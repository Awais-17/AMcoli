/**
 * @file amcoli-downloader.h
 * @brief Zero-dependency GGUF model downloader using system curl.
 *
 * Provides a model registry that maps short aliases (e.g. "qwen-3b") to
 * Hugging Face GGUF URLs, checks for local availability, and pulls models
 * automatically with a native curl progress bar.
 */

#ifndef AMCOLI_DOWNLOADER_H
#define AMCOLI_DOWNLOADER_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct amcoli_model_info {
    const char *alias;
    const char *name;
    const char *url;
    const char *filename;
    double      size_gb;
    double      total_params;  /* In billions */
    double      active_params; /* In billions */
    double      expert_size_mb;/* Estimated size of one expert */
};

/**
 * Get the list of all registered models.
 * Output count is written to count_out.
 */
const struct amcoli_model_info *amcoli_get_model_registry(int *count_out);

/**
 * Check if a model is registered under this alias.
 */
bool amcoli_model_is_registered(const char *alias);

/**
 * Get the full path where the model should reside locally.
 */
void amcoli_get_model_path(const char *alias, char *path_out, size_t max_len);

/**
 * Check if the model has already been downloaded.
 */
bool amcoli_model_exists(const char *alias);

/**
 * Download a registered model using system's native curl.
 * Prints progress bar to stderr.
 *
 * @param alias  Model registry alias
 * @return       true on success, false if download failed
 */
bool amcoli_download_model(const char *alias);

/**
 * Dry-run: print what would be downloaded without fetching any data.
 * @param alias  Model registry alias
 * @return       true if the alias is registered
 */
bool amcoli_dry_run(const char *alias);

/**
 * Verify a URL is reachable using curl range GET.
 * @param url        The URL to check
 * @param http_code  Receives the final HTTP status (200/206 = OK), may be NULL
 * @return           true if the URL resolves (HTTP 200 or 206)
 */
bool amcoli_check_url(const char *url, int *http_code);

/**
 * Check every registry URL and print a per-model report to stderr.
 * @return number of reachable models (out of g_model_count)
 */
int amcoli_check_registry(void);

/**
 * Check for newer AMcoli releases against the GitHub VERSION file.
 * Runs at most once per day (stamped in the OS temp dir) and prints a
 * one-line notice to stderr if a newer version is available. Never blocks
 * longer than the curl timeout. Safe to call from any thread.
 */
void amcoli_check_for_update(void);

#ifdef __cplusplus
}
#endif

#endif /* AMCOLI_DOWNLOADER_H */
