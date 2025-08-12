### Visualizer for Signal and Noise data

```sh
# To test locally
pip install -r requirements.txt
python src/main.py
gunicorn main:server -b 0.0.0.0:7860

# Then build and deploy!
docker build -t snr:latest .
docker run -p 7860:7860 snr:latest

# (push with git lfs)
git lfs install
git lfs track "*.json"
git lfs ls-files # check which files are tracked
```