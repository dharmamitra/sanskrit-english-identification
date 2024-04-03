# sanskrit-english-identification

use cargo run help to see explanation of commands

Train vectors --> cargo run train vectors --i1 "english/" --i2 "sanskrit/" -l "temp/" --l1 "english" --l2 "sanskrit"
Run predictions on vectors --> cargo run run predict-vectors -i model_english-sanskrit.bin -l <PATH TO FILES>