# Custom Running
The original multi-granularity test file is "best_ods_ois.py", from MuGE. 
For ease of use, I make a few changes to best_ods_ois.py and rename the file to python best_ods_ois_full.py. 
It can be conveniently used through commands
```
python best_ods_ois_full.py [single-scale dir]
```
Note:
1.  The "single-sacle test" folder cannot contain other nested folders. The file tree is as follows:
  ```
  single-scale dir
  |_-5.0
  |_-4.5
  |_...
  |_-0.5
  |_0.0
  ```
  There are a total of 11 edge folders of different granularities, and the names are fixed
  If the folder name needs to be replaced, the corresponding modification should be made on line 23 of best_ods_ois_full.py

2. Before multi-granularity testing, it is necessary to first complete single-granularity testing for the edges of each granularity respectively using Matlab.


