FROM continuumio/miniconda3

WORKDIR /src/earthkit-transforms

COPY environment.yml /src/earthkit-transforms/

RUN conda install -c conda-forge gcc python=3.10 \
    && conda env update -n base -f environment.yml

COPY . /src/earthkit-transforms

RUN pip install --no-deps -e .
