FROM continuumio/miniconda3

WORKDIR /src/earthkit-climate

COPY environment.yml /src/earthkit-climate/

RUN conda install -c conda-forge gcc python=3.10 \
    && conda env update -n base -f environment.yml

COPY . /src/earthkit-climate

RUN pip install --no-deps -e .
