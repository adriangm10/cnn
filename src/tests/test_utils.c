#include "test_utils.h"

#define __USE_GNU
#include <dlfcn.h>

void run_tests(test_t tests[], size_t test_count) {
  pid_t pid[test_count];
  Dl_info info[test_count];
  int wstatus[test_count];
  for (size_t i = 0; i < test_count; ++i) {
    dladdr(tests[i], &info[i]);
    printf("------- starting test: %s... -------\n\n", info[i].dli_sname);

    switch ((pid[i] = fork())) {
    case -1:
      perror("error running tests");
      break;
    case 0:
      tests[i]();
      exit(EXIT_SUCCESS);
    }
  }

  for (size_t i = 0; i < test_count; ++i) {
    waitpid(pid[i], &wstatus[i], 0);
  }

  puts("\n------- RESULTS ------\n");

  for (size_t i = 0; i < test_count; ++i) {
    if (WIFEXITED(wstatus[i])) {
      printf("test: %s... " GREEN " passed\n" RESET, info[i].dli_sname);
    } else {
      printf("test: %s... " RED " failed\n" RESET, info[i].dli_sname);
    }
  }
}
