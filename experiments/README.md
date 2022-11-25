

## XLA Flags for performance improvement on GPUs

```
export XLA_FLAGS='--xla_gpu_simplify_all_fp_conversions ${XLA_FLAGS}'
export XLA_FLAGS='--xla_gpu_all_reduce_combine_threshold_bytes=136314880 ${XLA_FLAGS}'
```

## Issues with accessing artifacts from open GCP Buckets

```
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
export TENSORSTORE_CA_BUNDLE="/etc/ssl/certs/ca-bundle.crt"
```
