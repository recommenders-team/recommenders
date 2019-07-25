# Copyright (C) 2012 Anaconda, Inc
# SPDX-License-Identifier: BSD-3-Clause

__conda_hashr() {
    if [ -n "${ZSH_VERSION:+x}" ]; then
        \rehash
    elif [ -n "${POSH_VERSION:+x}" ]; then
        :  # pass
    else
        \hash -r
    fi
}

__conda_activate() {
    if [ -n "${CONDA_PS1_BACKUP:+x}" ]; then
        # Handle transition from shell activated with conda <= 4.3 to a subsequent activation
        # after conda updated to >= 4.4. See issue #6173.
        PS1="$CONDA_PS1_BACKUP"
        \unset CONDA_PS1_BACKUP
    fi

    \local cmd="$1"
    shift
    \local ask_conda
    ask_conda="$(PS1="$PS1" "$CONDA_EXE" shell.posix "$cmd" "$@")" || \return $?
    \eval "$ask_conda"
    __conda_hashr
}

__conda_reactivate() {
    \local ask_conda
    ask_conda="$(PS1="$PS1" "$CONDA_EXE" shell.posix reactivate)" || \return $?
    \eval "$ask_conda"
    __conda_hashr
}

conda() {
    if [ "$#" -lt 1 ]; then
        "$CONDA_EXE"
    else
        \local cmd="$1"
        shift
        case "$cmd" in
            activate|deactivate)
                __conda_activate "$cmd" "$@"
                ;;
            install|update|upgrade|remove|uninstall)
                "$CONDA_EXE" "$cmd" "$@" && __conda_reactivate
                ;;
            *) "$CONDA_EXE" "$cmd" "$@" ;;
        esac
    fi
}

if [ -z "${CONDA_SHLVL+x}" ]; then
    \export CONDA_SHLVL=0
    PATH="$(dirname "$(dirname "$CONDA_EXE")")/condabin:${PATH:-}"
    \export PATH

    # We're not allowing PS1 to be unbound. It must at least be set.
    # However, we're not exporting it, which can cause problems when starting a second shell
    # via a first shell (i.e. starting zsh from bash).
    if [ -z "${PS1+x}" ]; then
        PS1=
    fi
fi
