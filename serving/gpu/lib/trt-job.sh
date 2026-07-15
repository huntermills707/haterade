# Shared helper for rendering the TensorRT engine-build Job YAML.
# Source this file from GPU serving scripts; do not execute it directly.

# render_trt_build_job emits a Kubernetes Job manifest that bakes a
# TensorRT plan inside the Triton 23.05 serving container.
#
# Args:
#   $1 job_name      (e.g. trt-build-gpu-v1)
#   $2 pvc_name      (e.g. triton-model-repo)
#   $3 model_name    (e.g. distilbert-toxicity)
#   $4 seq_len       (e.g. 128)
#   $5 max_batch     (e.g. 32)
render_trt_build_job() {
  local job_name="$1"
  local pvc_name="$2"
  local model_name="$3"
  local seq_len="$4"
  local max_batch="$5"

  cat <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${job_name}
  namespace: default
spec:
  template:
    metadata:
      labels:
        app: ${job_name}
    spec:
      restartPolicy: Never
      nodeSelector:
        nvidia.com/gpu.present: "true"
      containers:
        - name: trt
          image: nvcr.io/nvidia/tritonserver:23.05-py3
          command:
            - /usr/src/tensorrt/bin/trtexec
            - --onnx=/mnt/models/${model_name}/onnx/model.onnx
            - --saveEngine=/mnt/models/${model_name}/1/model.plan
            - --fp16
            - --minShapes=input_ids:1x${seq_len},attention_mask:1x${seq_len}
            - --optShapes=input_ids:${max_batch}x${seq_len},attention_mask:${max_batch}x${seq_len}
            - --maxShapes=input_ids:${max_batch}x${seq_len},attention_mask:${max_batch}x${seq_len}
          resources:
            limits:
              nvidia.com/gpu: 1
          volumeMounts:
            - name: model-repo
              mountPath: /mnt/models
      volumes:
        - name: model-repo
          persistentVolumeClaim:
            claimName: ${pvc_name}
EOF
}
