// Data parallel training loop. 
// 
// To run:
//     mpicc -O3 parallelisms/train_dp.c && mpirun -n <num-ranks> a.out

#include <mpi.h>
#include <stdlib.h>
#include "src/data.c"
#include "src/distributed.c"
#include "src/model.c"


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
    Model* model = Model_create(batch_size, seq_len, vocab_size, emb_size, hidden_size);

    // Train.
    float lr = 0.1;
    int steps = 25000;
    for (int step = 0; step < steps; step++) {
        Dataset_get_rank_batch(
            &train_split, global_Xs, global_Ys, Xs, Ys, global_batch_size, dist->dp_rank, dist->dp_size
        );
        float loss = Model_forward(model, Xs, Ys);
        allreduce_mean(&loss, /* size */1, dist->dp_comm, dist->dp_size);
        rank0_printf(dist->world_rank, "step: %d, loss %f\n", step, loss);

        Model_backward(model, Xs, Ys);
        allreduce_mean(model->wte->d_embedding, Embedding_numel(model->wte), dist->dp_comm, dist->dp_size);
        allreduce_mean(model->fc_1->d_weight, Linear_weight_numel(model->fc_1), dist->dp_comm, dist->dp_size);
        allreduce_mean(model->fc_1->d_bias, model->fc_1->out_features, dist->dp_comm, dist->dp_size);
        allreduce_mean(model->fc_2->d_weight, Linear_weight_numel(model->fc_2), dist->dp_comm, dist->dp_size);
        allreduce_mean(model->fc_2->d_bias, model->fc_2->out_features, dist->dp_comm, dist->dp_size);

        Model_step(model, lr);
    }

    // Validate.
    float loss = 0.0f;
    int n_valid_batches = 100;
    for (int i = 0; i < n_valid_batches; i ++) {
        Dataset_get_rank_batch(
            &test_split, global_Xs, global_Ys, Xs, Ys, global_batch_size, dist->dp_rank, dist->dp_size 
        );
        loss += Model_forward(model, Xs, Ys);
    }
    allreduce_mean(&loss, /* size */ 1, dist->dp_comm, dist->dp_size);
    rank0_printf(dist->world_rank, "Final validation loss: %f\n", loss / n_valid_batches);

    // Sample.
    if (dist->world_rank == 0) {
        int sample_batch_size = 1;
        int* sample_Xs = calloc(sizeof(float), batch_size * seq_len);
        int* dummy_Ys = calloc(sizeof(float), batch_size);
        for (int i = 0; i < 10; i++)  {
            Model_sample(model, sample_Xs, dummy_Ys, seq_len);
            Dataset_print_batch(sample_Xs, dummy_Ys, sample_batch_size, seq_len);
            memset(sample_Xs, 0, sizeof(float) * batch_size * seq_len);
        }
    }

    MPI_Finalize();
    return 0;
}
