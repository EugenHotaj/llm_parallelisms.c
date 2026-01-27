// Fully-sharded data parallel (i.e. Deepspeed ZeRO [1]) training loop. 
//
// Supports:
//     * Gradient sharding (i.e. ZeRO stage 2)
//     * Model parameter sharding (i.e. ZeRO stage 3)
// Optimizer parameter sharding is not (currently) supported because we use SGD.
// 
// To run:
//     mpicc -O3 parallelisms/train_dp.c && mpirun -n <num-ranks> a.out
//
// [1]: https://arxiv.org/abs/1910.02054


#include <mpi.h>
#include <stdlib.h>
#include "src/data.c"
#include "src/distributed.c"
#include "src/model.c"


float Model_forward_fsdp(
    Model* self, int* Xs, int* Ys, float* flat_buffer, MPI_Comm pg_comm, int pg_size
) {
    Embedding_forward_fsdp(self->wte, Xs, self->wte_out, flat_buffer, pg_comm, pg_size);
    Linear_forward_fsdp(self->fc_1, self->wte_out_flat, self->fc_1_out, flat_buffer, pg_comm, pg_size);
    relu(self->fc_1_out, self->relu_out);
    Linear_forward_fsdp(self->fc_2, self->relu_out, self->fc_2_out, flat_buffer, pg_comm, pg_size);
    softmax(self->fc_2_out, self->softmax_out);
    return cross_entropy_loss(self->softmax_out, Ys);
}


void Model_backward_fsdp(
    Model* self, int* Xs, int* Ys, float* flat_buffer, MPI_Comm pg_comm, int pg_size
) {
    Model_zerograd(self);
    cross_entropy_softmax_backward(self->fc_2_out, self->softmax_out, Ys);
    Linear_backward_fsdp(self->fc_2, self->relu_out, self->fc_2_out, flat_buffer, pg_comm, pg_size);
    relu_backward(self->fc_1_out, self->relu_out);
    Linear_backward_fsdp(self->fc_1, self->wte_out_flat, self->fc_1_out, flat_buffer, pg_comm, pg_size);
    Embedding_backward_fsdp(self->wte, Xs, self->wte_out, flat_buffer, pg_comm, pg_size);
}


void Model_sample_fsdp(
    Model* self, int* Xs, int* Ys, float* flat_buffer, MPI_Comm pg_comm, int pg_size, int seq_len
) {
    bool done = false;
    while (!done) {
        Model_forward_fsdp(self, Xs, Ys, flat_buffer, pg_comm, pg_size);
        int tok = Model_sample_token(self);
        // In theory, the model output and the RNG state should be identical across all ranks 
        // and hence "tok" should also be identical. However, this is not always the case 
        // in practice (possibly due to rank communication order) which can lead to MPI hangs 
        // if some ranks sample <BOS> before others. To overcome this issue, we broadcast the
        // sampled token from rank 0 to all other ranks.
        MPI_Bcast(&tok, /* count */ 1, MPI_INT, /* root */ 0, MPI_COMM_WORLD);
        done = Model_sample_update_input(Xs, Ys, tok, seq_len);
    }
}


int main(int argc, char** argv) {
    int global_batch_size = 32;
    int seq_len = 16;  // seq_len is computed offline and is equal to the longest word.
    int vocab_size = 27;
    int emb_size = 16;
    int hidden_size = 4 * emb_size;

    // Initialize environment. 
    srand(42);
    MPI_Init(&argc, &argv);
    int dp_size;
    MPI_Comm_size(MPI_COMM_WORLD, &dp_size);
    Dist* dist = Dist_create(/* tp_size */ 1, dp_size, /* pp_size */ 1);
    if (emb_size % dist->dp_size != 0) {
        rank0_printf(dist->world_rank, "Embedding size must be divisible by world size!\n");
        MPI_Finalize();
        exit(1);
    }
    if (hidden_size % dist->dp_size != 0) {
        rank0_printf(dist->world_rank, "Hidden dimension size must be divisible by world size!\n");
        MPI_Finalize();
        exit(1);
    }

    // Compute per-rank batch size from the global batch size.
    if (global_batch_size % dist->dp_size != 0) {
        rank0_printf(dist->world_rank, "Global batch size must be divisible by world size!\n");
        MPI_Finalize();
        exit(1);
    }
    int batch_size = global_batch_size / dist->dp_size;
    rank0_printf(dist->world_rank, "Micro batch_size: %d\n", batch_size); 

    // Create dataset.
    Dataset* dataset = Dataset_create_from_file("data/names.txt", seq_len);
    Dataset train_split, test_split;
    Dataset_train_test_split(dataset, &train_split, &test_split, /* train_percent */ 0.9);
    int* global_Xs = malloc(sizeof(int) * global_batch_size * seq_len);
    int* global_Ys = malloc(sizeof(int) * global_batch_size);
    int* Xs = malloc(sizeof(int) * batch_size * seq_len);
    int* Ys = malloc(sizeof(int) * batch_size);

    // Create model.
    // Hack! We first construct the full model then shard the parameters. This is just to 
    // ensure that the model parameters are initialized in the exact same way as the single-threaded
    // training loop for easy comparision. In practice, this approach would OOM for large models.
    Model* model = Model_create(batch_size, seq_len, vocab_size, emb_size, hidden_size);
    // Hack! We manually construct the padded embedding instead of using vocab_size_padded in
    // Model_create above. This ensures that the RNG state matches the single-threaded training
    // loop for easy comparison.
    Model_pad_vocab_fsdp(model, dist->dp_size);
    // Create temporary buffer to store allgathered params/grads of individual layers.
    float* flat_buffer = Model_create_flat_buffer_fsdp(model);
    Model_shard_fsdp(model, dist->dp_rank, dist->dp_size);

    // Train.
    float lr = 0.1f;
    int steps = 25000;
    for (int step = 0; step < steps; step++) {
        Dataset_get_rank_batch(
            &train_split, global_Xs, global_Ys, Xs, Ys, global_batch_size, dist->dp_rank, dist->dp_size 
        );
        float loss = Model_forward_fsdp(model, Xs, Ys, flat_buffer, dist->dp_comm, dist->dp_size);
        allreduce_mean(&loss, /* size */ 1, dist->dp_comm, dist->dp_size);
        rank0_printf(dist->world_rank, "step: %d, loss %f\n", step, loss);
        Model_backward_fsdp(model, Xs, Ys, flat_buffer, dist->dp_comm, dist->dp_size);
        Model_step(model, lr);
    }

    // Validate.
    float loss = 0.0f;
    int n_valid_batches = 100;
    for (int i = 0; i < n_valid_batches; i ++) {
        Dataset_get_rank_batch(
            &test_split, global_Xs, global_Ys, Xs, Ys, global_batch_size, dist->dp_rank, dist->dp_size 
        );
        loss += Model_forward_fsdp(model, Xs, Ys, flat_buffer, dist->dp_comm, dist->dp_size);
    }
    allreduce_mean(&loss, /* size */ 1, dist->dp_comm, dist->dp_size);
    rank0_printf(dist->world_rank, "Final validation loss: %f\n", loss / n_valid_batches);

    // Sample.
    int sample_batch_size = 1;
    int* sample_Xs = calloc(sizeof(float), batch_size * seq_len);
    int* dummy_Ys = calloc(sizeof(float), batch_size);
    for (int i = 0; i < 10 ; i++)  {
        Model_sample_fsdp(model, sample_Xs, dummy_Ys, flat_buffer, dist->dp_comm, dist->dp_size, seq_len);
        if (dist->world_rank == 0) {
            Dataset_print_batch(sample_Xs, dummy_Ys, sample_batch_size, seq_len);
        }
        memset(sample_Xs, 0, sizeof(float) * batch_size * seq_len);
    }

    MPI_Finalize();
    return 0;
}
