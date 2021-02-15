# python-probabilistic-examples

Hacking some Probabilistic Programming packages in Python

If you ever wanted to GPU using packages, make sure to install `libstdc++.so` with:

```
export LD_LIBRARY_PATH=$(nix eval --raw nixpkgs.stdenv.cc.cc.lib)/lib:$LD_LIBRARY_PATH
```
