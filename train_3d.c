// 3d Parallel training loop.
// 
// This is the culmination of our efforts which composes all implemented
// parallelisms. 
//
// As of late 2024, this is the state of the art in sharding dense (i.e. non-MoE),
// short-sequence (i.e. <32k) models and allows scaling to 400B+ parameters. For 
// example, Llama 3 405B [1] was pretrained using 3d parallelism. 
// 
// To run:
//     mpicc -O3 parallelisms/train_fsdp.c &&
//     mpirun -n <num-ranks> --map-by=:oversubscribe a.out --tp=<tp-ranks> --dp=<dp-ranks>
//
// [1]: https://arxiv.org/pdf/2407.21783


#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "src/data.c"
#include "src/distributed.c"
#include "src/model.c"


void Model_shard_3d(Model* self, Dist* dist) {
    // Note that the sharding order is important and should be done "inside->out".
    Model_shard_tp(self, dist->tp_rank, dist->tp_size);
    Model_shard_fsdp(self, dist->dp_rank, dist->dp_size);
    Model_shard_pp(self, dist->pp_rank);
}


float Model_forward_3d(Model* self, int* Xs, int* Ys, float* flat_buffer, Dist* dist) {
    float loss;
    if (dist->pp_rank == 0) {
        Embedding_forward_fsdp(self->wte, Xs, self->wte_out, flat_buffer, dist->dp_comm, dist->dp_size);
        send(self->wte_out->value, Activation_numel(self->wte_out), /* to_rank */ 1, dist->pp_comm);
    } else if (dist->pp_rank == 1) {
        recv(self->wte_out_flat->value, Activation_numel(self->wte_out_flat), /* from_rank */ 0, dist->pp_comm);
        Linear_forward_fsdp(self->fc_1, self->wte_out_flat, self->fc_1_out, flat_buffer, dist->dp_comm, dist->dp_size);
        relu(self->fc_1_out, self->relu_out);
        send(self->relu_out->value, Activation_numel(self->relu_out), /* to_rank */ 2, dist->pp_comm);
    } else if (dist->pp_rank == 2) {
        recv(self->relu_out->value, Activation_numel(self->relu_out), /* from_rank */ 1, dist->pp_comm);
        Linear_forward_fsdp(self->fc_2, self->relu_out, self->fc_2_out, flat_buffer, dist->dp_comm, dist->dp_size);
        allreduce_mean(self->fc_2_out->value, Activation_numel(self->fc_2_out), dist->tp_comm, dist->tp_size);
        softmax(self->fc_2_out, self->softmax_out);
        loss = cross_entropy_loss(self->softmax_out, Ys);
    } else {
        printf("Unknown pp_rank: %d\n", dist->pp_rank);
        MPI_Finalize();
        exit(1);
    }
    MPI_Bcast(&loss, /* count */ 1, MPI_FLOAT, /* root */ 2, dist->pp_comm);
    return loss;
}


void Model_backward_3d(Model* self, int* Xs, int* Ys, float* flat_buffer, Dist* dist) {
    Model_zerograd_pp(self, dist->pp_rank);
    if (dist->pp_rank == 2) {
        cross_entropy_softmax_backward(self->fc_2_out, self->softmax_out, Ys);
        Linear_backward_fsdp(self->fc_2, self->relu_out, self->fc_2_out, flat_buffer, dist->dp_comm, dist->dp_size);
        send(self->relu_out->d_value, Activation_numel(self->relu_out), /* to_rank */ 1, dist->pp_comm);
    } else if (dist->pp_rank == 1) {
        recv(self->relu_out->d_value, Activation_numel(self->relu_out), /* from_rank */ 2, dist->pp_comm);
        relu_backward(self->fc_1_out, self->relu_out);
        Linear_backward_fsdp(self->fc_1, self->wte_out_flat, self->fc_1_out, flat_buffer, dist->dp_comm, dist->dp_size);
        send(self->wte_out_flat->d_value, Activation_numel(self->wte_out_flat), /* to_rank */ 0, dist->pp_comm);
    } else if (dist->pp_rank == 0) {
        recv(self->wte_out->d_value, Activation_numel(self->wte_out), /* from_rank */ 1, dist->pp_comm);
        allreduce_mean(self->wte_out->d_value, Activation_numel(self->wte_out_flat), dist->tp_comm, dist->tp_size);
        Embedding_backward_fsdp(self->wte, Xs, self->wte_out, flat_buffer, dist->dp_comm, dist->dp_size);
    } else {
        printf("Unknown rank: %d\n", dist->pp_rank);
        MPI_Finalize();
        exit(1);
    }
}


void Model_sample_3d(Model* self, int* Xs, int* Ys, float* flat_buffer, Dist* dist, int seq_len) {
    bool done = false;
    while (!done) {
        Model_forward_3d(self, Xs, Ys, flat_buffer, dist);
        int tok;
        if (dist->pp_rank == 2) {
            tok = Model_sample_token(self);
        }
        // Choose an aribitrary pp_rank=2 and broadcast its token to the rest of the world. Because of
        // how we set up the distributed environment, we're guaranteed that the last rank is pp_rank=2.
        MPI_Bcast(&tok, /* count */ 1, MPI_INT, /* root */ dist->world_size - 1, MPI_COMM_WORLD);
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
    int tp_size, dp_size; 
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--tp") == 0) {
            tp_size = atoi(argv[i+1]);
        } else if (strcmp(argv[i], "--dp") == 0){
            dp_size = atoi(argv[i+1]);
        }
    }
    int pp_size = 3;  // Pipeline parallelism only supports 3 ranks.
    srand(42);
    MPI_Init(&argc, &argv);
    Dist* dist = Dist_create(tp_size, dp_size, pp_size);

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
    Model_shard_3d(model, dist);

    // Train.
    float lr = 0.1;
    int steps= 25000;
    for (int step = 0; step < steps; step++) {
        Dataset_get_rank_batch(
            &train_split, global_Xs, global_Ys, Xs, Ys, global_batch_size, dist->dp_rank, dist->dp_size 
        );
        float loss = Model_forward_3d(model, Xs, Ys, flat_buffer, dist);
        allreduce_mean(&loss, /* size */ 1, dist->dp_comm, dist->dp_size);
        rank0_printf(dist->world_rank, "step: %d, loss %f\n", step, loss);
        Model_backward_3d(model, Xs, Ys, flat_buffer, dist);
        Model_step_pp(model, lr, dist->pp_rank);
    }

    // Validate.
    float loss = 0.0f;
    int n_valid_batches = 100;
    for (int i = 0; i < n_valid_batches; i ++) {
        Dataset_get_rank_batch(
            &test_split, global_Xs, global_Ys, Xs, Ys, global_batch_size, dist->dp_rank, dist->dp_size 
        );
        loss += Model_forward_3d(model, Xs, Ys, flat_buffer, dist);
    }
    allreduce_mean(&loss, /* size */ 1, dist->dp_comm, dist->dp_size);
    rank0_printf(dist->world_rank, "Final validation loss: %f\n", loss / n_valid_batches);

    // Sample.
    int sample_batch_size = 1;
    int* sample_Xs = calloc(sizeof(float), batch_size * seq_len);
    int* dummy_Ys = calloc(sizeof(float), batch_size);
    for (int i = 0; i < 10 ; i++)  {
        Model_sample_3d(model, sample_Xs, dummy_Ys, flat_buffer, dist, seq_len);
        if (dist->world_rank == 0) {
            Dataset_print_batch(sample_Xs, dummy_Ys, sample_batch_size, seq_len);
        }
        memset(sample_Xs, 0, sizeof(float) * batch_size * seq_len);
    }

    MPI_Finalize();
    return 0;
}
