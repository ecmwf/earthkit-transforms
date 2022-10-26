FROM continuumio/miniconda3

WORKDIR /src/coucal

COPY environment.yml /src/coucal/

RUN conda install -c conda-forge gcc python=3.10 \
    && conda env update -n base -f environment.yml

COPY . /src/coucal

RUN pip install --no-deps -e .
