// Pipeline parallel training loop.
//
// To run:
//     mpicc -O3 parallelisms/train_pp.c && mpirun -n 3 a.out

#include <mpi.h>
#include <stdlib.h>
#include "src/data.c"
#include "src/distributed.c"
#include "src/model.c"


float Model_forward_pp(Model* self, int* Xs, int* Ys, int pg_rank, MPI_Comm pg_comm) {
    float loss;
    if (pg_rank == 0) {
        Embedding_forward(self->wte, Xs, self->wte_out);
        send(self->wte_out->value, Activation_numel(self->wte_out), /* to_rank */ 1, pg_comm);
    } else if (pg_rank == 1) {
        recv(self->wte_out_flat->value, Activation_numel(self->wte_out_flat), /* from_rank */ 0, pg_comm);
        Linear_forward(self->fc_1, self->wte_out_flat, self->fc_1_out);
        relu(self->fc_1_out, self->relu_out);
        send(self->relu_out->value, Activation_numel(self->relu_out), /* to_rank */ 2, pg_comm);
    } else if (pg_rank == 2) {
        recv(self->relu_out->value, Activation_numel(self->relu_out), /* from_rank */ 1, pg_comm);
        Linear_forward(self->fc_2, self->relu_out, self->fc_2_out);
        softmax(self->fc_2_out, self->softmax_out);
        loss = cross_entropy_loss(self->softmax_out, Ys);
    } else {
        printf("Unknown rank: %d\n", pg_rank);
        MPI_Finalize();
        exit(1);
    }
    // We don't technically need to broadcast here, but it's nicer if all the ranks have the
    // same loss value at the end.
    MPI_Bcast(&loss, /* count */ 1, MPI_FLOAT, /* root */ 2, MPI_COMM_WORLD);
    return loss;
}


void Model_backward_pp(Model* self, int* Xs, int* Ys, int pg_rank, MPI_Comm pg_comm) {
    Model_zerograd_pp(self, pg_rank);
    if (pg_rank == 2) {
        cross_entropy_softmax_backward(self->fc_2_out, self->softmax_out, Ys);
        Linear_backward(self->fc_2, self->relu_out, self->fc_2_out);
        send(self->relu_out->d_value, Activation_numel(self->relu_out), /* to_rank */ 1, pg_comm);
    } else if (pg_rank == 1) {
        recv(self->relu_out->d_value, Activation_numel(self->relu_out), /* from_rank */ 2, pg_comm);
        relu_backward(self->fc_1_out, self->relu_out);
        Linear_backward(self->fc_1, self->wte_out_flat, self->fc_1_out);
        send(self->wte_out_flat->d_value, Activation_numel(self->wte_out_flat), /* to_rank */ 0, pg_comm);
    } else if (pg_rank == 0) {
        recv(self->wte_out->d_value, Activation_numel(self->wte_out), /* from_rank */ 1, pg_comm);
        Embedding_backward(self->wte, Xs, self->wte_out);
    } else {
        printf("Unknown rank: %d\n", pg_rank);
        MPI_Finalize();
        exit(1);
    }
}


void Model_sample_pp(Model* self, int* Xs, int* Ys, int pg_rank, MPI_Comm pg_comm, int seq_len) {
    bool done = false;
    while (!done) {
        Model_forward_pp(self, Xs, Ys, pg_rank, pg_comm);
        int tok;
        if (pg_rank == 2) {
            tok = Model_sample_token(self);
        }
        MPI_Bcast(&tok, /* count */ 1, MPI_INT, /* root */ 2, MPI_COMM_WORLD);
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
    // Pipeline parallelism only supports 3 ranks.
    Dist* dist = Dist_create(/* tp_size */ 1, /* dp_size */ 1, /* pp_size */ 3);

    // Create dataset.
    Dataset* dataset = Dataset_create_from_file("data/names.txt", seq_len);
    Dataset train_split, test_split;
    Dataset_train_test_split(dataset, &train_split, &test_split, /* train_percent */ 0.9);
    int* Xs = malloc(sizeof(int) * batch_size * seq_len);
    int* Ys = malloc(sizeof(int) * batch_size);

    // Create model.
    // Hack! We first construct the full model then shard the parameters to stages. This is just to 
    // ensure that the model parameters are initialized in the exact same way as the single-threaded
    // training loop for easy comparision. In practice, this approach would OOM for large models.
    Model* model = Model_create(batch_size, seq_len, vocab_size, emb_size, hidden_size);
    Model_shard_pp(model, dist->pp_rank);

    // Train.
    float lr = 0.1;
    int steps = 25000;
    for (int step = 0; step < steps; step++) {
        Dataset_get_batch(&train_split, Xs, Ys, batch_size);
        float loss = Model_forward_pp(model, Xs, Ys, dist->pp_rank, dist->pp_comm);
        rank0_printf(dist->world_rank, "step: %d, loss %f\n", step, loss);
        Model_backward_pp(model, Xs, Ys, dist->pp_rank, dist->pp_comm);
        Model_step_pp(model, lr, dist->pp_rank);
    }

    // Validate.
    float loss = 0.0f;
    int n_valid_batches = 100;
    for (int i = 0; i < n_valid_batches; i ++) {
        Dataset_get_batch(&test_split, Xs, Ys, batch_size);
        loss += Model_forward_pp(model, Xs, Ys, dist->pp_rank, dist->pp_comm);
    }
    rank0_printf(dist->world_rank, "Final validation loss: %f\n", loss / n_valid_batches);

    // Sample.
    int sample_batch_size = 1;
    int* sample_Xs = calloc(sizeof(float), batch_size * seq_len);
    int* dummy_Ys = calloc(sizeof(float), batch_size);
    for (int i = 0; i < 10 ; i++)  {
        Model_sample_pp(model, sample_Xs, dummy_Ys, dist->pp_rank, dist->pp_comm, seq_len);
        if (dist->world_rank == 0) {
            Dataset_print_batch(sample_Xs, dummy_Ys, sample_batch_size, seq_len);
        }
        memset(sample_Xs, 0, sizeof(float) * batch_size * seq_len);
    }

    MPI_Finalize();
    return 0;
} 
