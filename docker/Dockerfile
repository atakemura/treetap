FROM continuumio/anaconda3:2021.11

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt --allow-releaseinfo-change update && \
    # python packages \
    conda update --all && \
    conda install category_encoders && \
    conda install -c conda-forge optuna && \
    conda install -c potassco clingo && \
    conda install -c conda-forge lightgbm && \
    conda install -c conda-forge shap

# install java
RUN mkdir -p /usr/share/man/man1 && \
    apt update && \
    apt install -y openjdk-11-jdk-headless

ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/

# weka dependencies
RUN apt install -y build-essential python3-dev && \
    conda install -c conda-forge javabridge && \
    pip install python-weka-wrapper3 && \
    conda install liac-arff

# rulefit
RUN pip install git+git://github.com/atakemura/rulefit.git@fix-pandas-error

# anchor
RUN pip install anchor-exp

RUN conda clean --all

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

RUN usermod -s /bin/bash user

CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]
