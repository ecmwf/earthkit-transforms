FROM continuumio/miniconda3

WORKDIR /src/tyto

COPY environment.yml /src/tyto/

RUN conda install -c conda-forge gcc python=3.10 \
    && conda env update -n base -f environment.yml

COPY . /src/tyto

RUN pip install --no-deps -e .
