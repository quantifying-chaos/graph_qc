# Intro to mathematica 

The functions uses square bracket, and the lists use braces.

```
in:= Range[5]
out:= {1,2,3,4,5}
```

There are some list manipulation and graphing functions

```
Join[{1, 2, 3}, {4, 5}, {6, 7}]
ListPlot[Reverse[Range[10]]]
ListLinePlot[{1, 3, 5, 4, 1, 2, 1, 4}]
NumberLinePlot[{1, 7, 11, 25}]
```

Operations can be applied to numerical list.

Note, there are not comment for mathematica. `#` is used here for convenience.

```
{1, 2, 3} + 10  # {11, 12, 13}
{1, 2, 3} + {10,1,2}  
```

The `Table[]` functions is the basic functionality for iteration and returning a list. 
There is no counterparts in common programming languages.

```
Table[RandomInteger[10], 20]  # {9, 1, 2, 3, 5}
```
