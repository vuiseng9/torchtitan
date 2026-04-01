setenv-cublaslt() {
    export CUBLASLT_LOG_LEVEL=2
    export CUBLASLT_LOG_FILE=log.cublaslt
    printenv CUBLASLT_LOG_LEVEL
    printenv CUBLASLT_LOG_FILE
}

unsetenv-cublaslt() {
    unset CUBLASLT_LOG_LEVEL
    unset CUBLASLT_LOG_FILE
    printenv CUBLASLT_LOG_LEVEL
    printenv CUBLASLT_LOG_FILE
}

rm-cublaslt-log() {
    # rm -f $CUBLASLT_LOG_FILE # for some reason, it doen't work, so hardcode the filename
    rm -f log.cublaslt
}

setenv-cublaslt