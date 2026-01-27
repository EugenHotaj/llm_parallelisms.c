// Tensor parallel (i.e. Megatron-LM [1]) training loop.
//
// To run:
//     mpicc -O3 parallelisms/train_tp.c && mpirun -n <num-ranks> a.out
//
// [1]: https://arxiv.org/abs/1909.08053


#include <mpi.h>
#include <stdlib.h>
#include "src/data.c"
#include "src/distributed.c"
#include "src/model.c"


float Model_forward_tp(Model* self, int* Xs, int* Ys, MPI_Comm pg_comm, int pg_size) {
    Embedding_forward(self->wte, Xs, self->wte_out);
    Linear_forward(self->fc_1, self->wte_out_flat, self->fc_1_out);
    relu(self->fc_1_out, self->relu_out);
    Linear_forward(self->fc_2, self->relu_out, self->fc_2_out);
    allreduce_mean(self->fc_2_out->value, Activation_numel(self->fc_2_out), pg_comm, pg_size);
    softmax(self->fc_2_out, self->softmax_out);
    return cross_entropy_loss(self->softmax_out, Ys);
}


void Model_backward_tp(Model* self, int* Xs, int* Ys, MPI_Comm pg_comm, int pg_size) {
    Model_zerograd(self);
    cross_entropy_softmax_backward(self->fc_2_out, self->softmax_out, Ys);
    Linear_backward(self->fc_2, self->relu_out, self->fc_2_out);
    relu_backward(self->fc_1_out, self->relu_out);
    Linear_backward(self->fc_1, self->wte_out_flat, self->fc_1_out);
    allreduce_mean(self->wte_out_flat->d_value, Activation_numel(self->wte_out_flat), pg_comm, pg_size);
    Embedding_backward(self->wte, Xs, self->wte_out);
}


void Model_sample_tp(Model* self, int* Xs, int* Ys, MPI_Comm pg_comm, int pg_size, int seq_len) {
    bool done = false;
    while (!done) {
        Model_forward_tp(self, Xs, Ys, pg_comm, pg_size);
        int tok = Model_sample_token(self);
        done = Model_sample_update_input(Xs, Ys, tok, seq_len);
    }
}


int main(int argc, char** argv) {
    int batch_size = 32;
    int seq_len = 16;  // seq_len is computed offline and is equal to the longest word.
    int vocab_size = 27;
    int emb_size = 16;
    int hidden_size = 4 * emb_size;

    // Initialize environment. 
    srand(42);
    MPI_Init(&argc, &argv);
    int tp_size;
    MPI_Comm_size(MPI_COMM_WORLD, &tp_size);
    Dist* dist = Dist_create(tp_size, /* dp_size */ 1, /* pp_size */ 1);
    if (hidden_size % dist->tp_size != 0) {
        rank0_printf(dist->world_rank, "Hidden size must be divisible by world size!\n");
        MPI_Finalize();
        exit(1);
    }

    // Create dataset.
    Dataset* dataset = Dataset_create_from_file("data/names.txt", seq_len);
    Dataset train_split, test_split;
    Dataset_train_test_split(dataset, &train_split, &test_split, /* train_percent */ 0.9);
    int* Xs = malloc(sizeof(int) * batch_size * seq_len);
    int* Ys = malloc(sizeof(int) * batch_size);

    // Create model.
    // Hack! We first construct the full model then shard the parameters. This is just to 
    // ensure that the model parameters are initialized in the exact same way as the single-threaded
    // training loop for easy comparision. In practice, this approach would OOM for large models.
    Model* model = Model_create(batch_size, seq_len, vocab_size, emb_size, hidden_size);
    Model_shard_tp(model, dist->tp_rank, dist->tp_size);

    float lr = 0.1;
    int steps = 25000;
    for (int step = 0; step < steps; step++) {
        Dataset_get_batch(&train_split, Xs, Ys, batch_size);
        float loss = Model_forward_tp(model, Xs, Ys, dist->tp_comm, dist->tp_size);
        rank0_printf(dist->world_rank, "step: %d, loss %f\n", step, loss);
        Model_backward_tp(model, Xs, Ys, dist->tp_comm, dist->tp_size);
        Model_step(model, lr);
    }

    // Validate.
    float loss = 0.0f;
    int n_valid_batches = 100;
    for (int i = 0; i < n_valid_batches; i ++) {
        Dataset_get_batch(&test_split, Xs, Ys, batch_size);
        loss += Model_forward_tp(model, Xs, Ys, dist->tp_comm, dist->tp_size);
    }
    rank0_printf(dist->world_rank, "Final validation loss: %f\n", loss / n_valid_batches);

    // Sample.
    int sample_batch_size = 1;
    int* sample_Xs = calloc(sizeof(float), batch_size * seq_len);
    int* dummy_Ys = calloc(sizeof(float), batch_size);
    for (int i = 0; i < 10 ; i++)  {
        Model_sample_tp(model, sample_Xs, dummy_Ys, dist->tp_comm, dist->tp_size, seq_len);
        if (dist->world_rank == 0) {
            Dataset_print_batch(sample_Xs, dummy_Ys, sample_batch_size, seq_len);
        }
        memset(sample_Xs, 0, sizeof(float) * batch_size * seq_len);
    }

    MPI_Finalize();
    return 0;
}
