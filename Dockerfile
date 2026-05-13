FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    numpy>=1.24 \
    pandas>=2.0 \
    scipy>=1.11 \
    scikit-learn>=1.3 \
    pyyaml \
    joblib \
    tqdm \
    pyarrow

COPY aikixp/ aikixp/
COPY scripts/predict.py scripts/predict.py
COPY configs/ configs/

ENTRYPOINT ["python", "scripts/predict.py"]
