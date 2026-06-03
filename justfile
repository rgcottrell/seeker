# seeker container image — build & run the Vulkan inference server in Podman.
# Multistage Containerfile: rust:bookworm builder -> debian:bookworm-slim + Mesa RADV.

image := "seeker"
tag   := "latest"
port  := "11434"
# Host directory holding .gguf model files; mounted read-only at /models.
models := justfile_directory() / "models"

# qwen35moe model for `just chat`: resolved offline from the host Hugging Face
# cache (mounted read-only). `hf_cache` is the host HF_HOME; the repo/file pin
# the unsloth Qwen3.6-35B-A3B (Q4_K_XL) GGUF without hardcoding a snapshot hash.
hf_cache  := "/models/huggingface"
qwen_repo := "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
qwen_file := "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"

# Build the container image.
build:
    podman build -t {{image}}:{{tag}} -f Containerfile {{justfile_directory()}}

# Run the server with full AMD GPU passthrough; the port is published 1:1 on the host.
#   just serve                                  # no model (only /health + /apply-template)
#   just serve qwen3.gguf                        # --model /models/qwen3.gguf
#   just serve qwen3.gguf --ctx-size 8192 --parallel 4
serve model="" *extra="":
    podman run --rm -it \
        --device /dev/dri \
        --group-add keep-groups \
        --security-opt label=disable \
        -p {{port}}:{{port}} \
        -v {{models}}:/models:ro \
        {{image}}:{{tag}} \
        serve --host 0.0.0.0 --port {{port}} \
        {{ if model != "" { "--model /models/" + model } else { "" } }} \
        {{extra}}

# Interactive chat REPL against the qwen35moe model with full AMD GPU passthrough.
# The model is resolved offline from the host HF cache (mounted read-only), so no
# download happens. Extra args pass through to `seeker chat`:
#   just chat                          # start the REPL
#   just chat --ctx-size 8192 --temp 0.7
chat *extra="":
    podman run --rm -it \
        --device /dev/dri \
        --group-add keep-groups \
        --security-opt label=disable \
        -v {{hf_cache}}:{{hf_cache}}:ro \
        -e HF_HOME={{hf_cache}} \
        {{image}}:{{tag}} \
        chat --hf-repo {{qwen_repo}} --hf-file {{qwen_file}} --offline \
        {{extra}}
