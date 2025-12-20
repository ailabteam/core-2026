uv venv
source .venv/bin/activate
<!-- uv pip install -U vllm --torch-backend auto -->


source venv/bin/activate

-> for Deepseek ORC
pip install flash-attn==2.7.3 --no-build-isolation


pip install -r requirements.txt

python -m src.pipeline data-test/4_1.jpg --output outputs