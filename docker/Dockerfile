FROM continuumio/anaconda3:latest

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt update && \
    # python packages
    conda install pip && \
    pip install --upgrade pip && \
    pip install problog && \
    pip install category_encoders && \
    pip install optuna && \
    conda install -c potassco clingo && \
    conda install -c potassco asprin && \
    conda install -c conda-forge lightgbm && \
    conda install -c conda-forge xgboost && \
    conda install -c conda-forge shap

# until this is merged https://github.com/potassco/asprin/pull/3
# RUN sed -i -e 's/async/async_/g' /opt/conda/lib/python3.7/site-packages/asprin/src/utils/clingo_signal_handler.py

#    && \
#    conda install -c r r &&
#    R -e 'install.packages(c("inTrees", "randomForest", "nodeHarvest"), repos="https://cran.ism.ac.jp")'
#
#
#    # clingocon build dependencies
#    apt install -y build-essential re2c bison && \
#    ln -s /usr/include/locale.h /usr/include/xlocale.h && \
#    mkdir -p /tmp/clingcon && \
#    cd /tmp/clingcon && \
#    git clone https://github.com/potassco/clingcon.git ./ && \
#    mkdir build && \
#    git submodule update --init --recursive && \
#    cmake -H./clingo/ -B./build/ -DCMAKE_BUILD_TYPE=Release && \
#    # NB. by default this clingcon modified clingo will NOT be found before the original clingo.
#    # If you want to use this clingcon modify the PATH
#    cmake --build /tmp/clingcon/build -j 4 --target install
