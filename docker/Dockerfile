FROM continuumio/anaconda3:2021.11

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt --allow-releaseinfo-change update && \
    # python packages \
    conda update --all && \
    conda install -c potassco clingo

RUN pip install category_encoders optuna lightgbm shap

# install java
RUN mkdir -p /usr/share/man/man1 && \
    apt update && \
    apt install -y openjdk-11-jdk-headless

ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/

# weka dependencies
RUN apt install -y build-essential python3-dev

RUN conda install -c conda-forge javabridge && \
    pip install python-weka-wrapper3 && \
    conda install liac-arff

# rulefit
RUN pip install git+https://github.com/atakemura/rulefit.git@fix-pandas-error

# anchor
RUN pip install anchor-exp

RUN conda clean --all

# pyxai with dependencies
RUN echo 'deb http://deb.debian.org/debian bullseye-backports main' >> /etc/apt/sources.list && \
    apt update && \
    apt install -y openssh-server tzdata graphviz build-essential && \
    apt install -y ffmpeg libsm6 libxext6 qt6-base-dev libxcb-cursor0 libxcb-xinerama0 nodejs && \
    pip install pyxai

# remote development
RUN apt install -y openssh-server

RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion \
  && mkdir /run/sshd

RUN useradd -m user \
  && yes password | passwd user

RUN mkdir /home/user/workspace && \
    chown -R user:user /home/user/workspace

USER user

WORKDIR /home/user/workspace
RUN git clone https://github.com/atakemura/treetap.git

USER root

RUN usermod -s /bin/bash user

CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]
