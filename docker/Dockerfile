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
#
#ARG USERNAME=akihiro
#ARG USER_UID=1000
#ARG USER_GID=$USER_UID
#
#RUN addgroup --gid ${USER_GID} ${USERNAME}
#RUN adduser --disabled-password --gecos "" --uid ${USER_UID} --gid ${USER_GID} ${USERNAME}
#RUN chown -R ${USERNAME} /home/${USERNAME}
#ENV HOME /home/${USERNAME}
#
#USER ${USERNAME}

RUN mkdir -p /usr/share/man/man1 && \
    apt update && \
    apt install -y openjdk-11-jdk-headless

ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/

RUN apt install -y build-essential python3-dev && \
    pip install javabridge && \
    pip install python-weka-wrapper3 && \
    pip install liac-arff

#ADD https://prdownloads.sourceforge.net/weka/weka-3-8-4.zip ./

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
