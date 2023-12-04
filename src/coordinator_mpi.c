#include <mpi.h>

#include "coordinator.h"

#define READY 0
#define NEW_TASK 1
#define TERMINATE -1
#define TASK_COMPLETED 2

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Error: not enough arguments\n");
    printf("Usage: %s [path_to_task_list]\n", argv[0]);
    return -1;
  }

  // TODO: implement Open MPI coordinator
  
  int procID, totalProcs;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procID);
  int numTasks;
  task_t **tasks;
  if(read_tasks(argv[1], &numTasks, &tasks)) return -1;



  if (procID == 0) {
      
      int nextTask = 0;
      MPI_Status status;
      int32_t message;
      
      while (nextTask < numTasks) {
          
          MPI_Recv(&message, 1, MPI_INT32_T, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
          int source = status.MPI_SOURCE;
          MPI_Send(&nextTask, 1, MPI_INT32_T, source, 0, MPI_COMM_WORLD);
          nextTask++;
      }
      int32_t termin = TERMINATE;
      for (int i = 1; i < totalProcs; ++i) {
          MPI_Recv(&message, 1, MPI_INT32_T, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
          int source = status.MPI_SOURCE;
          MPI_Send(&termin, 1, MPI_INT32_T, source, 0, MPI_COMM_WORLD);
          
      
      } 
  } else {
      task_t task;
          int32_t message;
          int32_t ready = READY;
          
          while (true) {
              
              MPI_Status status;
              MPI_Send(&message, 1, MPI_INT32_T, 0, 0, MPI_COMM_WORLD);
              MPI_Recv(&message, 1, MPI_INT32_T, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              if (message == TERMINATE) {
                  break;
              } else {
                  if (execute_task(tasks[message])) {
                      return -1;
                  }
                  free(tasks[message]->path);
              }
          }
  }
  free(tasks);
  MPI_Finalize();
  return 0;

}
