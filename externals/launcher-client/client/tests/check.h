// A minimal test harness: CHECK records a failure and goes on; main returns the count.
#ifndef LAUNCHER_CLIENT_TESTS_CHECK_H
#define LAUNCHER_CLIENT_TESTS_CHECK_H

#include <cstdio>

inline int g_failures = 0;

#define CHECK(cond)                                                                \
	do {                                                                           \
		if (!(cond)) {                                                             \
			std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
			g_failures++;                                                          \
		}                                                                          \
	} while (0)

#define FAIL(...)                                          \
	do {                                                   \
		std::fprintf(stderr, "%s:%d: ", __FILE__, __LINE__); \
		std::fprintf(stderr, __VA_ARGS__);                 \
		std::fprintf(stderr, "\n");                        \
		g_failures++;                                      \
	} while (0)

inline int finish(const char* name) {
	if (g_failures == 0) {
		std::printf("%s: ok\n", name);
	} else {
		std::printf("%s: %d failure(s)\n", name, g_failures);
	}
	return g_failures == 0 ? 0 : 1;
}

#endif // LAUNCHER_CLIENT_TESTS_CHECK_H
