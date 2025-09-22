Image Annotation Service (FastAPI + Ollama + Qwen2.5-VL)

Quickstart
1) Setup
```bash
source /data/shivanvitha/ai-toolkit/aitool/bin/activate
```

2) Start Ollama (in another terminal)
```bash
ollama serve
```

3) Run the server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
or 
```bash
sh run_server.sh
```

4) Health check
```bash
curl http://localhost:8000/health
```

5) Describe an image
```bash
curl -X POST http://localhost:8000/qwen_api \
  -F 'prompt=Describe the image in detail.' \
  -F 'file=@sample.jpg'
```

6) Optional: with a reference image
```bash
curl -X POST http://localhost:8000/qwen_api \
  -F 'prompt=Compare to the reference.' \
  -F 'file=@user.jpg' \
  -F 'ref_file=@reference.jpg'
```

Client (batch directory)
```bash
source .venv/bin/activate
python client.py --dir ./samples --token "(([AA] man))" 
```
By default the desciption files will be stores in the --dir path. If you want another location use --out tag

Client (Running from Remote) 
Clone the client.py file and change the Url for API_URL to
```bash 
API_URL = "http://99.50.230.172:8000/qwen_api"  
```
Notes
- Default model in code: qwen2.5vl:32b
- Ollama endpoint used: http://localhost:11434/api/generate

