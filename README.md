# STL_to_LS_CUDA

Convert STL to Level-set function.

## Example

See [main.cu](./main.cu).

Note: To run [main.cu](./main.cu), compile

```
make
```

and run the generated executable file `a.out`:

```
./a.out
```

In this example, the level-set function is calculated from the STL file [Stanford_Bunny.stl](./Stanford_Bunny.stl) (downloaded from [here](https://commons.wikimedia.org/wiki/File:Stanford_Bunny.stl?uselang=ja)) and output the result to the file `bunny.vtk`.
The results can be visualized in paraview.

<img src="./figs/example_bunny.png" width = 80%>

## Reference

[1]: [(Qiita) バイナリフォーマットstlファイルをC++で読み込む, @JmpM (まるや)](https://qiita.com/JmpM/items/9355c655614d47b5d67b)

[2] [Mittal et al., A versatile sharp interface immersed boundary method for incompressible flows with complex boundaries., Journal of Computational Physics 227 (2008) 4825-4852,](https://www.sciencedirect.com/science/article/pii/S0021999108000235)

 [3] [(Qiita) ParaViewでVTKレガシーフォーマットを使う その1, @kaityo256 (ロボ太)](https://qiita.com/kaityo256/items/661833e9e2bfbac31d4b)