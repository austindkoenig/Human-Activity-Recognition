# <center>Activity Recognizer Output Log</center>

## Preprocessing

```
{
    "train_split": 0.85,
    "test_split": 0.15,
    "shapes": {
        "x_train": [
            713,
            1024,
            3
        ],
        "y_train": [
            713,
            14
        ],
        "x_test": [
            126,
            1024,
            3
        ],
        "y_test": [
            126,
            14
        ]
    }
}
```
## Model Hierarchy
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d_1 (Conv1D)            (None, 1021, 512)         6656      
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 1018, 512)         1049088   
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 1015, 512)         1049088   
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 126, 512)          0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 126, 512)          0         
_________________________________________________________________
conv1d_4 (Conv1D)            (None, 123, 512)          1049088   
_________________________________________________________________
conv1d_5 (Conv1D)            (None, 120, 512)          1049088   
_________________________________________________________________
conv1d_6 (Conv1D)            (None, 117, 512)          1049088   
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 19, 512)           0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 19, 512)           0         
_________________________________________________________________
conv1d_7 (Conv1D)            (None, 18, 512)           524800    
_________________________________________________________________
conv1d_8 (Conv1D)            (None, 17, 512)           524800    
_________________________________________________________________
conv1d_9 (Conv1D)            (None, 16, 512)           524800    
_________________________________________________________________
max_pooling1d_3 (MaxPooling1 (None, 4, 512)            0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 4, 512)            0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2048)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              2098176   
_________________________________________________________________
dense_2 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
dense_3 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
dropout_4 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_4 (Dense)              (None, 14)                14350     
=================================================================
Total params: 11,038,222
Trainable params: 11,038,222
Non-trainable params: 0
_________________________________________________________________
```
## Model Training

```
Train on 570 samples, validate on 143 samples
Epoch 1/256
 - 14s - loss: 65.6111 - acc: 0.1088 - val_loss: 63.7461 - val_acc: 0.0979

Epoch 00001: val_acc improved from -inf to 0.09790, saving model to ./output/model.hdf5
Epoch 2/256
 - 6s - loss: 62.0006 - acc: 0.1316 - val_loss: 60.2336 - val_acc: 0.1049

Epoch 00002: val_acc improved from 0.09790 to 0.10490, saving model to ./output/model.hdf5
Epoch 3/256
 - 6s - loss: 58.6252 - acc: 0.1070 - val_loss: 56.9617 - val_acc: 0.1049

Epoch 00003: val_acc did not improve from 0.10490
Epoch 4/256
 - 6s - loss: 55.4866 - acc: 0.1246 - val_loss: 53.9330 - val_acc: 0.1678

Epoch 00004: val_acc improved from 0.10490 to 0.16783, saving model to ./output/model.hdf5
Epoch 5/256
 - 6s - loss: 52.5174 - acc: 0.1632 - val_loss: 50.9990 - val_acc: 0.2238

Epoch 00005: val_acc improved from 0.16783 to 0.22378, saving model to ./output/model.hdf5
Epoch 6/256
 - 6s - loss: 49.6504 - acc: 0.2526 - val_loss: 48.1223 - val_acc: 0.3497

Epoch 00006: val_acc improved from 0.22378 to 0.34965, saving model to ./output/model.hdf5
Epoch 7/256
 - 7s - loss: 46.7049 - acc: 0.3526 - val_loss: 45.1658 - val_acc: 0.4056

Epoch 00007: val_acc improved from 0.34965 to 0.40559, saving model to ./output/model.hdf5
Epoch 8/256
 - 6s - loss: 43.8727 - acc: 0.4895 - val_loss: 42.3180 - val_acc: 0.5874

Epoch 00008: val_acc improved from 0.40559 to 0.58741, saving model to ./output/model.hdf5
Epoch 9/256
 - 7s - loss: 41.3407 - acc: 0.5702 - val_loss: 40.1077 - val_acc: 0.6713

Epoch 00009: val_acc improved from 0.58741 to 0.67133, saving model to ./output/model.hdf5
Epoch 10/256
 - 6s - loss: 39.0605 - acc: 0.5684 - val_loss: 37.7443 - val_acc: 0.6643

Epoch 00010: val_acc did not improve from 0.67133
Epoch 11/256
 - 6s - loss: 36.8786 - acc: 0.6070 - val_loss: 35.8555 - val_acc: 0.5944

Epoch 00011: val_acc did not improve from 0.67133
Epoch 12/256
 - 7s - loss: 34.8406 - acc: 0.6281 - val_loss: 33.7457 - val_acc: 0.6993

Epoch 00012: val_acc improved from 0.67133 to 0.69930, saving model to ./output/model.hdf5
Epoch 13/256
 - 6s - loss: 32.9143 - acc: 0.6649 - val_loss: 32.0615 - val_acc: 0.6853

Epoch 00013: val_acc did not improve from 0.69930
Epoch 14/256
 - 6s - loss: 31.1499 - acc: 0.6596 - val_loss: 30.1232 - val_acc: 0.7483

Epoch 00014: val_acc improved from 0.69930 to 0.74825, saving model to ./output/model.hdf5
Epoch 15/256
 - 20s - loss: 29.4387 - acc: 0.6649 - val_loss: 28.4369 - val_acc: 0.7483

Epoch 00015: val_acc did not improve from 0.74825
Epoch 16/256
 - 28s - loss: 27.8926 - acc: 0.6544 - val_loss: 27.0231 - val_acc: 0.6993

Epoch 00016: val_acc did not improve from 0.74825
Epoch 17/256
 - 21s - loss: 26.3712 - acc: 0.6965 - val_loss: 25.5929 - val_acc: 0.6993

Epoch 00017: val_acc did not improve from 0.74825
Epoch 18/256
 - 20s - loss: 24.9558 - acc: 0.6947 - val_loss: 24.4848 - val_acc: 0.6294

Epoch 00018: val_acc did not improve from 0.74825
Epoch 19/256
 - 20s - loss: 23.6408 - acc: 0.6842 - val_loss: 22.9979 - val_acc: 0.7133

Epoch 00019: val_acc did not improve from 0.74825
Epoch 20/256
 - 20s - loss: 22.3214 - acc: 0.7140 - val_loss: 21.5927 - val_acc: 0.7622

Epoch 00020: val_acc improved from 0.74825 to 0.76224, saving model to ./output/model.hdf5
Epoch 21/256
 - 20s - loss: 21.2164 - acc: 0.6965 - val_loss: 21.0259 - val_acc: 0.5664

Epoch 00021: val_acc did not improve from 0.76224
Epoch 22/256
 - 20s - loss: 20.0356 - acc: 0.7281 - val_loss: 19.4721 - val_acc: 0.7622

Epoch 00022: val_acc did not improve from 0.76224
Epoch 23/256
 - 61s - loss: 19.0285 - acc: 0.6596 - val_loss: 18.5470 - val_acc: 0.6783

Epoch 00023: val_acc did not improve from 0.76224
Epoch 24/256
 - 64s - loss: 18.0234 - acc: 0.7105 - val_loss: 17.4867 - val_acc: 0.6853

Epoch 00024: val_acc did not improve from 0.76224
Epoch 25/256
 - 68s - loss: 17.0377 - acc: 0.7123 - val_loss: 16.6246 - val_acc: 0.7692

Epoch 00025: val_acc improved from 0.76224 to 0.76923, saving model to ./output/model.hdf5
Epoch 26/256
 - 67s - loss: 16.1908 - acc: 0.6912 - val_loss: 15.6311 - val_acc: 0.7972

Epoch 00026: val_acc improved from 0.76923 to 0.79720, saving model to ./output/model.hdf5
Epoch 27/256
 - 64s - loss: 15.3196 - acc: 0.7386 - val_loss: 14.8040 - val_acc: 0.7902

Epoch 00027: val_acc did not improve from 0.79720
Epoch 28/256
 - 62s - loss: 14.5536 - acc: 0.7000 - val_loss: 14.1357 - val_acc: 0.7343

Epoch 00028: val_acc did not improve from 0.79720
Epoch 29/256
 - 66s - loss: 13.7451 - acc: 0.7263 - val_loss: 13.3375 - val_acc: 0.7343

Epoch 00029: val_acc did not improve from 0.79720
Epoch 30/256
 - 56s - loss: 13.0466 - acc: 0.7333 - val_loss: 12.7346 - val_acc: 0.7063

Epoch 00030: val_acc did not improve from 0.79720
Epoch 31/256
 - 55s - loss: 12.3627 - acc: 0.7263 - val_loss: 11.9384 - val_acc: 0.7902

Epoch 00031: val_acc did not improve from 0.79720
Epoch 32/256
 - 56s - loss: 11.7545 - acc: 0.7281 - val_loss: 11.3838 - val_acc: 0.7483

Epoch 00032: val_acc did not improve from 0.79720
Epoch 33/256
 - 57s - loss: 11.1684 - acc: 0.7158 - val_loss: 10.9301 - val_acc: 0.7483

Epoch 00033: val_acc did not improve from 0.79720
Epoch 34/256
 - 58s - loss: 10.5923 - acc: 0.7351 - val_loss: 10.3484 - val_acc: 0.7413

Epoch 00034: val_acc did not improve from 0.79720
Epoch 35/256
 - 57s - loss: 10.0820 - acc: 0.7368 - val_loss: 9.9571 - val_acc: 0.6503

Epoch 00035: val_acc did not improve from 0.79720
Epoch 36/256
 - 27s - loss: 9.5395 - acc: 0.7421 - val_loss: 9.2676 - val_acc: 0.7832

Epoch 00036: val_acc did not improve from 0.79720
Epoch 37/256
 - 18s - loss: 9.0728 - acc: 0.7246 - val_loss: 8.7733 - val_acc: 0.7692

Epoch 00037: val_acc did not improve from 0.79720
Epoch 38/256
 - 18s - loss: 8.6479 - acc: 0.7368 - val_loss: 8.4282 - val_acc: 0.7343

Epoch 00038: val_acc did not improve from 0.79720
Epoch 39/256
 - 18s - loss: 8.2076 - acc: 0.7333 - val_loss: 7.9884 - val_acc: 0.7762

Epoch 00039: val_acc did not improve from 0.79720
Epoch 40/256
 - 20s - loss: 7.8054 - acc: 0.7404 - val_loss: 7.7413 - val_acc: 0.7413

Epoch 00040: val_acc did not improve from 0.79720
Epoch 41/256
 - 18s - loss: 7.4188 - acc: 0.7456 - val_loss: 7.4237 - val_acc: 0.7273

Epoch 00041: val_acc did not improve from 0.79720
Epoch 42/256
 - 17s - loss: 7.0784 - acc: 0.7421 - val_loss: 6.9025 - val_acc: 0.7343

Epoch 00042: val_acc did not improve from 0.79720
Epoch 43/256
 - 18s - loss: 6.7651 - acc: 0.7211 - val_loss: 6.5541 - val_acc: 0.7552

Epoch 00043: val_acc did not improve from 0.79720
Epoch 44/256
 - 20s - loss: 6.3581 - acc: 0.7474 - val_loss: 6.3021 - val_acc: 0.7483

Epoch 00044: val_acc did not improve from 0.79720
Epoch 45/256
 - 18s - loss: 6.0951 - acc: 0.7526 - val_loss: 5.9866 - val_acc: 0.7692

Epoch 00045: val_acc did not improve from 0.79720
Epoch 46/256
 - 18s - loss: 5.8002 - acc: 0.7754 - val_loss: 5.7060 - val_acc: 0.7832

Epoch 00046: val_acc did not improve from 0.79720
Epoch 47/256
 - 18s - loss: 5.5776 - acc: 0.7526 - val_loss: 5.5229 - val_acc: 0.7203

Epoch 00047: val_acc did not improve from 0.79720
Epoch 48/256
 - 18s - loss: 5.3019 - acc: 0.7561 - val_loss: 5.3370 - val_acc: 0.7273

Epoch 00048: val_acc did not improve from 0.79720
Epoch 49/256
 - 19s - loss: 5.0580 - acc: 0.7579 - val_loss: 5.1631 - val_acc: 0.7133

Epoch 00049: val_acc did not improve from 0.79720
Epoch 50/256
 - 18s - loss: 4.8264 - acc: 0.7702 - val_loss: 4.8465 - val_acc: 0.7622

Epoch 00050: val_acc did not improve from 0.79720
Epoch 51/256
 - 17s - loss: 4.6136 - acc: 0.7614 - val_loss: 4.5806 - val_acc: 0.7692

Epoch 00051: val_acc did not improve from 0.79720
Epoch 52/256
 - 17s - loss: 4.4035 - acc: 0.7439 - val_loss: 4.5071 - val_acc: 0.7203

Epoch 00052: val_acc did not improve from 0.79720
Epoch 53/256
 - 17s - loss: 4.1818 - acc: 0.7860 - val_loss: 4.2196 - val_acc: 0.7692

Epoch 00053: val_acc did not improve from 0.79720
Epoch 54/256
 - 15s - loss: 4.0617 - acc: 0.7772 - val_loss: 3.9947 - val_acc: 0.7762

Epoch 00054: val_acc did not improve from 0.79720
Epoch 55/256
 - 17s - loss: 3.8639 - acc: 0.7649 - val_loss: 3.9036 - val_acc: 0.7692

Epoch 00055: val_acc did not improve from 0.79720
Epoch 56/256
 - 19s - loss: 3.7072 - acc: 0.7526 - val_loss: 3.7270 - val_acc: 0.7832

Epoch 00056: val_acc did not improve from 0.79720
Epoch 57/256
 - 19s - loss: 3.5559 - acc: 0.7614 - val_loss: 3.5122 - val_acc: 0.7692

Epoch 00057: val_acc did not improve from 0.79720
Epoch 58/256
 - 19s - loss: 3.4090 - acc: 0.7789 - val_loss: 3.3703 - val_acc: 0.7972

Epoch 00058: val_acc did not improve from 0.79720
Epoch 59/256
 - 16s - loss: 3.2572 - acc: 0.7667 - val_loss: 3.5021 - val_acc: 0.6783

Epoch 00059: val_acc did not improve from 0.79720
Epoch 60/256
 - 18s - loss: 3.1089 - acc: 0.8035 - val_loss: 3.2847 - val_acc: 0.7203

Epoch 00060: val_acc did not improve from 0.79720
Epoch 61/256
 - 23s - loss: 2.9831 - acc: 0.7842 - val_loss: 3.1998 - val_acc: 0.7203

Epoch 00061: val_acc did not improve from 0.79720
Epoch 62/256
 - 19s - loss: 2.9563 - acc: 0.7544 - val_loss: 3.7935 - val_acc: 0.5874

Epoch 00062: val_acc did not improve from 0.79720
Epoch 63/256
 - 17s - loss: 2.8424 - acc: 0.7404 - val_loss: 2.8695 - val_acc: 0.7832

Epoch 00063: val_acc did not improve from 0.79720
Epoch 64/256
 - 18s - loss: 2.7031 - acc: 0.7719 - val_loss: 2.7045 - val_acc: 0.8042

Epoch 00064: val_acc improved from 0.79720 to 0.80420, saving model to ./output/model.hdf5
Epoch 65/256
 - 16s - loss: 2.6130 - acc: 0.7877 - val_loss: 2.8456 - val_acc: 0.6993

Epoch 00065: val_acc did not improve from 0.80420
Epoch 66/256
 - 19s - loss: 2.4912 - acc: 0.8018 - val_loss: 2.7075 - val_acc: 0.7692

Epoch 00066: val_acc did not improve from 0.80420
Epoch 67/256
 - 18s - loss: 2.4421 - acc: 0.7719 - val_loss: 2.4662 - val_acc: 0.7622

Epoch 00067: val_acc did not improve from 0.80420
Epoch 68/256
 - 18s - loss: 2.3721 - acc: 0.7789 - val_loss: 2.3677 - val_acc: 0.7552

Epoch 00068: val_acc did not improve from 0.80420
Epoch 69/256
 - 17s - loss: 2.2884 - acc: 0.7754 - val_loss: 2.3554 - val_acc: 0.7622

Epoch 00069: val_acc did not improve from 0.80420
Epoch 70/256
 - 18s - loss: 2.2330 - acc: 0.7684 - val_loss: 2.3180 - val_acc: 0.7552

Epoch 00070: val_acc did not improve from 0.80420
Epoch 71/256
 - 20s - loss: 2.1451 - acc: 0.7789 - val_loss: 2.2748 - val_acc: 0.7343

Epoch 00071: val_acc did not improve from 0.80420
Epoch 72/256
 - 18s - loss: 2.0924 - acc: 0.7789 - val_loss: 2.2039 - val_acc: 0.7552

Epoch 00072: val_acc did not improve from 0.80420
Epoch 73/256
 - 18s - loss: 2.0112 - acc: 0.7825 - val_loss: 2.1183 - val_acc: 0.7832

Epoch 00073: val_acc did not improve from 0.80420
Epoch 74/256
 - 6s - loss: 1.9728 - acc: 0.7912 - val_loss: 2.2926 - val_acc: 0.7273

Epoch 00074: val_acc did not improve from 0.80420
Epoch 75/256
 - 6s - loss: 1.8946 - acc: 0.7877 - val_loss: 2.2627 - val_acc: 0.6573

Epoch 00075: val_acc did not improve from 0.80420
Epoch 76/256
 - 5s - loss: 1.8358 - acc: 0.7789 - val_loss: 1.9076 - val_acc: 0.7762

Epoch 00076: val_acc did not improve from 0.80420
Epoch 77/256
 - 6s - loss: 1.7881 - acc: 0.7825 - val_loss: 1.9956 - val_acc: 0.7622

Epoch 00077: val_acc did not improve from 0.80420
Epoch 78/256
 - 7s - loss: 1.7775 - acc: 0.7842 - val_loss: 1.8333 - val_acc: 0.7902

Epoch 00078: val_acc did not improve from 0.80420
Epoch 79/256
 - 7s - loss: 1.7424 - acc: 0.7702 - val_loss: 2.0268 - val_acc: 0.7343

Epoch 00079: val_acc did not improve from 0.80420
Epoch 80/256
 - 6s - loss: 1.6676 - acc: 0.7912 - val_loss: 1.8887 - val_acc: 0.7203

Epoch 00080: val_acc did not improve from 0.80420
Epoch 81/256
 - 7s - loss: 1.6257 - acc: 0.7737 - val_loss: 1.6059 - val_acc: 0.7902

Epoch 00081: val_acc did not improve from 0.80420
Epoch 82/256
 - 6s - loss: 1.5559 - acc: 0.7930 - val_loss: 1.6467 - val_acc: 0.7413

Epoch 00082: val_acc did not improve from 0.80420
Epoch 83/256
 - 6s - loss: 1.5799 - acc: 0.7807 - val_loss: 1.8197 - val_acc: 0.7483

Epoch 00083: val_acc did not improve from 0.80420
Epoch 84/256
 - 6s - loss: 1.5388 - acc: 0.7754 - val_loss: 1.5962 - val_acc: 0.7832

Epoch 00084: val_acc did not improve from 0.80420
Epoch 85/256
 - 6s - loss: 1.4877 - acc: 0.7702 - val_loss: 1.7329 - val_acc: 0.7133

Epoch 00085: val_acc did not improve from 0.80420
Epoch 86/256
 - 7s - loss: 1.4416 - acc: 0.7930 - val_loss: 1.6352 - val_acc: 0.7902

Epoch 00086: val_acc did not improve from 0.80420
Epoch 87/256
 - 6s - loss: 1.4455 - acc: 0.7684 - val_loss: 1.5420 - val_acc: 0.7622

Epoch 00087: val_acc did not improve from 0.80420
Epoch 88/256
 - 6s - loss: 1.3789 - acc: 0.7982 - val_loss: 1.6594 - val_acc: 0.7762

Epoch 00088: val_acc did not improve from 0.80420
Epoch 89/256
 - 7s - loss: 1.3556 - acc: 0.8053 - val_loss: 1.5248 - val_acc: 0.7552

Epoch 00089: val_acc did not improve from 0.80420
Epoch 90/256
 - 7s - loss: 1.3589 - acc: 0.7930 - val_loss: 1.5330 - val_acc: 0.7692

Epoch 00090: val_acc did not improve from 0.80420
Epoch 91/256
 - 6s - loss: 1.3883 - acc: 0.7807 - val_loss: 1.4752 - val_acc: 0.7622

Epoch 00091: val_acc did not improve from 0.80420
Epoch 92/256
 - 6s - loss: 1.2975 - acc: 0.8000 - val_loss: 1.5246 - val_acc: 0.7692

Epoch 00092: val_acc did not improve from 0.80420
Epoch 93/256
 - 6s - loss: 1.2498 - acc: 0.8000 - val_loss: 1.5582 - val_acc: 0.7413

Epoch 00093: val_acc did not improve from 0.80420
Epoch 94/256
 - 7s - loss: 1.2781 - acc: 0.8000 - val_loss: 1.5270 - val_acc: 0.7483

Epoch 00094: val_acc did not improve from 0.80420
Epoch 95/256
 - 6s - loss: 1.2482 - acc: 0.7807 - val_loss: 1.4385 - val_acc: 0.7902

Epoch 00095: val_acc did not improve from 0.80420
Epoch 96/256
 - 6s - loss: 1.2290 - acc: 0.8140 - val_loss: 1.9244 - val_acc: 0.6573

Epoch 00096: val_acc did not improve from 0.80420
Epoch 97/256
 - 6s - loss: 1.2268 - acc: 0.8000 - val_loss: 1.3387 - val_acc: 0.7972

Epoch 00097: val_acc did not improve from 0.80420
Epoch 98/256
 - 6s - loss: 1.1999 - acc: 0.7895 - val_loss: 1.4706 - val_acc: 0.7622

Epoch 00098: val_acc did not improve from 0.80420
Epoch 99/256
 - 6s - loss: 1.1535 - acc: 0.8123 - val_loss: 1.9968 - val_acc: 0.6573

Epoch 00099: val_acc did not improve from 0.80420
Epoch 100/256
 - 6s - loss: 1.1979 - acc: 0.8000 - val_loss: 1.3226 - val_acc: 0.8042

Epoch 00100: val_acc did not improve from 0.80420
Epoch 101/256
 - 6s - loss: 1.1154 - acc: 0.7965 - val_loss: 1.3729 - val_acc: 0.7413

Epoch 00101: val_acc did not improve from 0.80420
Epoch 102/256
 - 6s - loss: 1.1444 - acc: 0.8000 - val_loss: 2.0874 - val_acc: 0.6084

Epoch 00102: val_acc did not improve from 0.80420
Epoch 103/256
 - 6s - loss: 1.1573 - acc: 0.7947 - val_loss: 1.3268 - val_acc: 0.7832

Epoch 00103: val_acc did not improve from 0.80420
Epoch 104/256
 - 6s - loss: 1.1766 - acc: 0.7825 - val_loss: 1.3153 - val_acc: 0.7762

Epoch 00104: val_acc did not improve from 0.80420
Epoch 105/256
 - 6s - loss: 1.1157 - acc: 0.8018 - val_loss: 1.3609 - val_acc: 0.7483

Epoch 00105: val_acc did not improve from 0.80420
Epoch 106/256
 - 6s - loss: 1.1661 - acc: 0.7807 - val_loss: 1.3818 - val_acc: 0.7692

Epoch 00106: val_acc did not improve from 0.80420
Epoch 107/256
 - 6s - loss: 1.1424 - acc: 0.7930 - val_loss: 1.3344 - val_acc: 0.7622

Epoch 00107: val_acc did not improve from 0.80420
Epoch 108/256
 - 6s - loss: 1.1098 - acc: 0.8053 - val_loss: 1.1913 - val_acc: 0.8182

Epoch 00108: val_acc improved from 0.80420 to 0.81818, saving model to ./output/model.hdf5
Epoch 109/256
 - 7s - loss: 1.0542 - acc: 0.8088 - val_loss: 1.3585 - val_acc: 0.7762

Epoch 00109: val_acc did not improve from 0.81818
Epoch 110/256
 - 7s - loss: 1.1346 - acc: 0.7895 - val_loss: 1.3792 - val_acc: 0.7413

Epoch 00110: val_acc did not improve from 0.81818
Epoch 111/256
 - 7s - loss: 1.0862 - acc: 0.8105 - val_loss: 1.5194 - val_acc: 0.6783

Epoch 00111: val_acc did not improve from 0.81818
Epoch 112/256
 - 7s - loss: 1.0943 - acc: 0.8035 - val_loss: 1.3308 - val_acc: 0.7552

Epoch 00112: val_acc did not improve from 0.81818
Epoch 113/256
 - 6s - loss: 1.0720 - acc: 0.8000 - val_loss: 1.1423 - val_acc: 0.8042

Epoch 00113: val_acc did not improve from 0.81818
Epoch 114/256
 - 6s - loss: 1.0462 - acc: 0.8035 - val_loss: 1.7250 - val_acc: 0.6154

Epoch 00114: val_acc did not improve from 0.81818
Epoch 115/256
 - 7s - loss: 1.0849 - acc: 0.7860 - val_loss: 1.1201 - val_acc: 0.7692

Epoch 00115: val_acc did not improve from 0.81818
Epoch 116/256
 - 7s - loss: 0.9996 - acc: 0.8000 - val_loss: 1.3141 - val_acc: 0.7622

Epoch 00116: val_acc did not improve from 0.81818
Epoch 117/256
 - 6s - loss: 1.0469 - acc: 0.7877 - val_loss: 1.2123 - val_acc: 0.7692

Epoch 00117: val_acc did not improve from 0.81818
Epoch 118/256
 - 5s - loss: 1.0150 - acc: 0.8158 - val_loss: 1.2877 - val_acc: 0.7832

Epoch 00118: val_acc did not improve from 0.81818
Epoch 119/256
 - 5s - loss: 1.0500 - acc: 0.7912 - val_loss: 1.3571 - val_acc: 0.7622

Epoch 00119: val_acc did not improve from 0.81818
Epoch 120/256
 - 7s - loss: 1.0297 - acc: 0.8158 - val_loss: 1.4356 - val_acc: 0.7343

Epoch 00120: val_acc did not improve from 0.81818
Epoch 121/256
 - 6s - loss: 1.0137 - acc: 0.7930 - val_loss: 1.4361 - val_acc: 0.7273

Epoch 00121: val_acc did not improve from 0.81818
Epoch 122/256
 - 6s - loss: 1.0662 - acc: 0.7930 - val_loss: 1.1073 - val_acc: 0.8042

Epoch 00122: val_acc did not improve from 0.81818
Epoch 123/256
 - 6s - loss: 0.9823 - acc: 0.8175 - val_loss: 1.2195 - val_acc: 0.7622

Epoch 00123: val_acc did not improve from 0.81818
Epoch 124/256
 - 6s - loss: 1.0890 - acc: 0.7807 - val_loss: 1.1998 - val_acc: 0.7902

Epoch 00124: val_acc did not improve from 0.81818
Epoch 125/256
 - 6s - loss: 0.9810 - acc: 0.8053 - val_loss: 1.4125 - val_acc: 0.7692

Epoch 00125: val_acc did not improve from 0.81818
Epoch 126/256
 - 6s - loss: 1.0182 - acc: 0.7982 - val_loss: 1.3687 - val_acc: 0.7832

Epoch 00126: val_acc did not improve from 0.81818
Epoch 127/256
 - 6s - loss: 0.9957 - acc: 0.8018 - val_loss: 1.1296 - val_acc: 0.8042

Epoch 00127: val_acc did not improve from 0.81818
Epoch 128/256
 - 6s - loss: 0.9764 - acc: 0.8053 - val_loss: 1.2811 - val_acc: 0.7832

Epoch 00128: val_acc did not improve from 0.81818
Epoch 129/256
 - 6s - loss: 0.9796 - acc: 0.8105 - val_loss: 1.3043 - val_acc: 0.7692

Epoch 00129: val_acc did not improve from 0.81818
Epoch 130/256
 - 6s - loss: 0.9629 - acc: 0.8088 - val_loss: 1.3651 - val_acc: 0.7413

Epoch 00130: val_acc did not improve from 0.81818
Epoch 131/256
 - 6s - loss: 0.9838 - acc: 0.8018 - val_loss: 1.2057 - val_acc: 0.7622

Epoch 00131: val_acc did not improve from 0.81818
Epoch 132/256
 - 6s - loss: 0.9727 - acc: 0.8211 - val_loss: 1.2696 - val_acc: 0.7273

Epoch 00132: val_acc did not improve from 0.81818
Epoch 133/256
 - 6s - loss: 0.9965 - acc: 0.7947 - val_loss: 1.4445 - val_acc: 0.7133

Epoch 00133: val_acc did not improve from 0.81818
Epoch 134/256
 - 7s - loss: 0.9194 - acc: 0.8035 - val_loss: 1.0798 - val_acc: 0.8322

Epoch 00134: val_acc improved from 0.81818 to 0.83217, saving model to ./output/model.hdf5
Epoch 135/256
 - 6s - loss: 0.9188 - acc: 0.8368 - val_loss: 2.8473 - val_acc: 0.4545

Epoch 00135: val_acc did not improve from 0.83217
Epoch 136/256
 - 6s - loss: 1.0276 - acc: 0.7860 - val_loss: 1.1290 - val_acc: 0.7902

Epoch 00136: val_acc did not improve from 0.83217
Epoch 137/256
 - 6s - loss: 0.9818 - acc: 0.8263 - val_loss: 1.3601 - val_acc: 0.7413

Epoch 00137: val_acc did not improve from 0.83217
Epoch 138/256
 - 6s - loss: 0.9973 - acc: 0.8070 - val_loss: 1.3716 - val_acc: 0.7622

Epoch 00138: val_acc did not improve from 0.83217
Epoch 139/256
 - 6s - loss: 0.9594 - acc: 0.8123 - val_loss: 1.2147 - val_acc: 0.7273

Epoch 00139: val_acc did not improve from 0.83217
Epoch 140/256
 - 7s - loss: 0.9168 - acc: 0.8333 - val_loss: 1.1403 - val_acc: 0.7902

Epoch 00140: val_acc did not improve from 0.83217
Epoch 141/256
 - 6s - loss: 0.9070 - acc: 0.8298 - val_loss: 1.2358 - val_acc: 0.7832

Epoch 00141: val_acc did not improve from 0.83217
Epoch 142/256
 - 6s - loss: 0.9618 - acc: 0.8351 - val_loss: 1.2581 - val_acc: 0.7203

Epoch 00142: val_acc did not improve from 0.83217
Epoch 143/256
 - 6s - loss: 0.9873 - acc: 0.7807 - val_loss: 1.4256 - val_acc: 0.7552

Epoch 00143: val_acc did not improve from 0.83217
Epoch 144/256
 - 7s - loss: 0.9763 - acc: 0.8175 - val_loss: 1.3568 - val_acc: 0.7483

Epoch 00144: val_acc did not improve from 0.83217
Epoch 145/256
 - 6s - loss: 0.9654 - acc: 0.8123 - val_loss: 1.2133 - val_acc: 0.7902

Epoch 00145: val_acc did not improve from 0.83217
Epoch 146/256
 - 6s - loss: 0.9774 - acc: 0.8123 - val_loss: 1.6760 - val_acc: 0.6014

Epoch 00146: val_acc did not improve from 0.83217
Epoch 147/256
 - 6s - loss: 0.9653 - acc: 0.8105 - val_loss: 1.1774 - val_acc: 0.7692

Epoch 00147: val_acc did not improve from 0.83217
Epoch 148/256
 - 6s - loss: 0.9476 - acc: 0.7982 - val_loss: 1.2766 - val_acc: 0.7413

Epoch 00148: val_acc did not improve from 0.83217
Epoch 149/256
 - 6s - loss: 0.9344 - acc: 0.8193 - val_loss: 1.1987 - val_acc: 0.7832

Epoch 00149: val_acc did not improve from 0.83217
Epoch 150/256
 - 6s - loss: 0.9357 - acc: 0.8263 - val_loss: 1.1922 - val_acc: 0.7483

Epoch 00150: val_acc did not improve from 0.83217
Epoch 151/256
 - 6s - loss: 0.9563 - acc: 0.8018 - val_loss: 1.1059 - val_acc: 0.8042

Epoch 00151: val_acc did not improve from 0.83217
Epoch 152/256
 - 6s - loss: 0.9621 - acc: 0.7982 - val_loss: 1.0694 - val_acc: 0.7832

Epoch 00152: val_acc did not improve from 0.83217
Epoch 153/256
 - 6s - loss: 0.9312 - acc: 0.8211 - val_loss: 1.2383 - val_acc: 0.7552

Epoch 00153: val_acc did not improve from 0.83217
Epoch 154/256
 - 6s - loss: 0.9684 - acc: 0.8123 - val_loss: 1.1851 - val_acc: 0.7832

Epoch 00154: val_acc did not improve from 0.83217
Epoch 155/256
 - 7s - loss: 0.9124 - acc: 0.8140 - val_loss: 1.1984 - val_acc: 0.8042

Epoch 00155: val_acc did not improve from 0.83217
Epoch 156/256
 - 5s - loss: 0.9210 - acc: 0.8421 - val_loss: 1.3744 - val_acc: 0.7762

Epoch 00156: val_acc did not improve from 0.83217
Epoch 157/256
 - 6s - loss: 0.9432 - acc: 0.8298 - val_loss: 1.2107 - val_acc: 0.7972

Epoch 00157: val_acc did not improve from 0.83217
Epoch 158/256
 - 7s - loss: 0.9429 - acc: 0.8193 - val_loss: 1.0824 - val_acc: 0.8042

Epoch 00158: val_acc did not improve from 0.83217
Epoch 159/256
 - 6s - loss: 0.9293 - acc: 0.8228 - val_loss: 1.3367 - val_acc: 0.6993

Epoch 00159: val_acc did not improve from 0.83217
Epoch 160/256
 - 6s - loss: 0.9284 - acc: 0.8070 - val_loss: 1.5979 - val_acc: 0.6783

Epoch 00160: val_acc did not improve from 0.83217
Epoch 161/256
 - 5s - loss: 0.9546 - acc: 0.8140 - val_loss: 1.2237 - val_acc: 0.7972

Epoch 00161: val_acc did not improve from 0.83217
Epoch 162/256
 - 6s - loss: 0.9311 - acc: 0.8158 - val_loss: 1.1572 - val_acc: 0.8252

Epoch 00162: val_acc did not improve from 0.83217
Epoch 163/256
 - 6s - loss: 0.9367 - acc: 0.8263 - val_loss: 1.4093 - val_acc: 0.6573

Epoch 00163: val_acc did not improve from 0.83217
Epoch 164/256
 - 6s - loss: 0.9408 - acc: 0.8088 - val_loss: 1.2948 - val_acc: 0.7483

Epoch 00164: val_acc did not improve from 0.83217
Epoch 165/256
 - 6s - loss: 0.9017 - acc: 0.8228 - val_loss: 1.3379 - val_acc: 0.7483

Epoch 00165: val_acc did not improve from 0.83217
Epoch 166/256
 - 6s - loss: 0.9034 - acc: 0.8263 - val_loss: 1.3485 - val_acc: 0.7762

Epoch 00166: val_acc did not improve from 0.83217
Epoch 167/256
 - 6s - loss: 0.9153 - acc: 0.8246 - val_loss: 1.1975 - val_acc: 0.8042

Epoch 00167: val_acc did not improve from 0.83217
Epoch 168/256
 - 6s - loss: 0.9056 - acc: 0.8140 - val_loss: 1.2236 - val_acc: 0.7692

Epoch 00168: val_acc did not improve from 0.83217
Epoch 169/256
 - 6s - loss: 0.9400 - acc: 0.8053 - val_loss: 1.1566 - val_acc: 0.7832

Epoch 00169: val_acc did not improve from 0.83217
Epoch 170/256
 - 6s - loss: 0.9092 - acc: 0.8263 - val_loss: 1.1555 - val_acc: 0.7972

Epoch 00170: val_acc did not improve from 0.83217
Epoch 171/256
 - 6s - loss: 0.9185 - acc: 0.8404 - val_loss: 1.3101 - val_acc: 0.7902

Epoch 00171: val_acc did not improve from 0.83217
Epoch 172/256
 - 6s - loss: 0.9267 - acc: 0.8193 - val_loss: 1.4937 - val_acc: 0.7203

Epoch 00172: val_acc did not improve from 0.83217
Epoch 173/256
 - 6s - loss: 0.8775 - acc: 0.8719 - val_loss: 1.1920 - val_acc: 0.7972

Epoch 00173: val_acc did not improve from 0.83217
Epoch 174/256
 - 7s - loss: 0.9254 - acc: 0.8158 - val_loss: 1.0937 - val_acc: 0.8042

Epoch 00174: val_acc did not improve from 0.83217
Epoch 175/256
 - 6s - loss: 0.9207 - acc: 0.8316 - val_loss: 1.3539 - val_acc: 0.7552

Epoch 00175: val_acc did not improve from 0.83217
Epoch 176/256
 - 7s - loss: 0.9897 - acc: 0.8035 - val_loss: 1.3700 - val_acc: 0.6923

Epoch 00176: val_acc did not improve from 0.83217
Epoch 177/256
 - 7s - loss: 1.0061 - acc: 0.8088 - val_loss: 1.3050 - val_acc: 0.7692

Epoch 00177: val_acc did not improve from 0.83217
Epoch 178/256
 - 6s - loss: 0.9689 - acc: 0.8228 - val_loss: 1.3222 - val_acc: 0.7483

Epoch 00178: val_acc did not improve from 0.83217
Epoch 179/256
 - 6s - loss: 0.8826 - acc: 0.8474 - val_loss: 1.1683 - val_acc: 0.8042

Epoch 00179: val_acc did not improve from 0.83217
Epoch 180/256
 - 7s - loss: 0.9296 - acc: 0.8333 - val_loss: 1.5263 - val_acc: 0.6923

Epoch 00180: val_acc did not improve from 0.83217
Epoch 181/256
 - 6s - loss: 0.9582 - acc: 0.8193 - val_loss: 1.4286 - val_acc: 0.7483

Epoch 00181: val_acc did not improve from 0.83217
Epoch 182/256
 - 6s - loss: 0.9247 - acc: 0.8351 - val_loss: 1.3484 - val_acc: 0.7832

Epoch 00182: val_acc did not improve from 0.83217
Epoch 183/256
 - 6s - loss: 0.9481 - acc: 0.8298 - val_loss: 1.1914 - val_acc: 0.7692

Epoch 00183: val_acc did not improve from 0.83217
Epoch 184/256
 - 6s - loss: 0.9184 - acc: 0.8298 - val_loss: 1.3544 - val_acc: 0.8042

Epoch 00184: val_acc did not improve from 0.83217
Epoch 185/256
 - 5s - loss: 0.9129 - acc: 0.8368 - val_loss: 1.4417 - val_acc: 0.7972

Epoch 00185: val_acc did not improve from 0.83217
Epoch 186/256
 - 6s - loss: 0.9217 - acc: 0.8281 - val_loss: 1.1351 - val_acc: 0.8112

Epoch 00186: val_acc did not improve from 0.83217
Epoch 187/256
 - 6s - loss: 0.9036 - acc: 0.8386 - val_loss: 1.1418 - val_acc: 0.7972

Epoch 00187: val_acc did not improve from 0.83217
Epoch 188/256
 - 6s - loss: 0.9076 - acc: 0.8474 - val_loss: 1.2935 - val_acc: 0.7692

Epoch 00188: val_acc did not improve from 0.83217
Epoch 189/256
 - 6s - loss: 0.8746 - acc: 0.8421 - val_loss: 1.3454 - val_acc: 0.7273

Epoch 00189: val_acc did not improve from 0.83217
Epoch 190/256
 - 6s - loss: 0.9383 - acc: 0.8158 - val_loss: 1.1440 - val_acc: 0.7972

Epoch 00190: val_acc did not improve from 0.83217
Epoch 191/256
 - 6s - loss: 0.8723 - acc: 0.8351 - val_loss: 1.2582 - val_acc: 0.8042

Epoch 00191: val_acc did not improve from 0.83217
Epoch 192/256
 - 6s - loss: 0.9227 - acc: 0.8386 - val_loss: 3.0374 - val_acc: 0.4196

Epoch 00192: val_acc did not improve from 0.83217
Epoch 193/256
 - 6s - loss: 1.0321 - acc: 0.7982 - val_loss: 1.3658 - val_acc: 0.7483

Epoch 00193: val_acc did not improve from 0.83217
Epoch 194/256
 - 6s - loss: 0.8924 - acc: 0.8316 - val_loss: 1.3047 - val_acc: 0.8042

Epoch 00194: val_acc did not improve from 0.83217
Epoch 195/256
 - 6s - loss: 0.8789 - acc: 0.8386 - val_loss: 1.1031 - val_acc: 0.8042

Epoch 00195: val_acc did not improve from 0.83217
Epoch 196/256
 - 7s - loss: 0.9047 - acc: 0.8439 - val_loss: 1.2675 - val_acc: 0.7762

Epoch 00196: val_acc did not improve from 0.83217
Epoch 197/256
 - 6s - loss: 0.9598 - acc: 0.8246 - val_loss: 1.4409 - val_acc: 0.7483

Epoch 00197: val_acc did not improve from 0.83217
Epoch 198/256
 - 6s - loss: 0.9693 - acc: 0.8088 - val_loss: 1.2309 - val_acc: 0.8042

Epoch 00198: val_acc did not improve from 0.83217
Epoch 199/256
 - 6s - loss: 0.9015 - acc: 0.8368 - val_loss: 1.5158 - val_acc: 0.7552

Epoch 00199: val_acc did not improve from 0.83217
Epoch 200/256
 - 6s - loss: 0.9145 - acc: 0.8316 - val_loss: 1.2441 - val_acc: 0.8182

Epoch 00200: val_acc did not improve from 0.83217
Epoch 201/256
 - 6s - loss: 0.9054 - acc: 0.8579 - val_loss: 1.3334 - val_acc: 0.7133

Epoch 00201: val_acc did not improve from 0.83217
Epoch 202/256
 - 7s - loss: 0.8992 - acc: 0.8386 - val_loss: 1.2048 - val_acc: 0.8182

Epoch 00202: val_acc did not improve from 0.83217
Epoch 203/256
 - 6s - loss: 0.8865 - acc: 0.8404 - val_loss: 1.1291 - val_acc: 0.8112

Epoch 00203: val_acc did not improve from 0.83217
Epoch 204/256
 - 6s - loss: 0.9371 - acc: 0.8211 - val_loss: 1.0803 - val_acc: 0.7902

Epoch 00204: val_acc did not improve from 0.83217
Epoch 205/256
 - 7s - loss: 0.8862 - acc: 0.8439 - val_loss: 1.5002 - val_acc: 0.6783

Epoch 00205: val_acc did not improve from 0.83217
Epoch 206/256
 - 6s - loss: 0.9416 - acc: 0.8088 - val_loss: 1.2750 - val_acc: 0.7832

Epoch 00206: val_acc did not improve from 0.83217
Epoch 207/256
 - 6s - loss: 0.9300 - acc: 0.8228 - val_loss: 1.2448 - val_acc: 0.7832

Epoch 00207: val_acc did not improve from 0.83217
Epoch 208/256
 - 6s - loss: 0.8475 - acc: 0.8421 - val_loss: 1.2509 - val_acc: 0.8182

Epoch 00208: val_acc did not improve from 0.83217
Epoch 209/256
 - 6s - loss: 0.8926 - acc: 0.8404 - val_loss: 1.2791 - val_acc: 0.7902

Epoch 00209: val_acc did not improve from 0.83217
Epoch 210/256
 - 6s - loss: 0.8828 - acc: 0.8561 - val_loss: 1.3021 - val_acc: 0.8042

Epoch 00210: val_acc did not improve from 0.83217
Epoch 211/256
 - 6s - loss: 0.9469 - acc: 0.8211 - val_loss: 1.1643 - val_acc: 0.8042

Epoch 00211: val_acc did not improve from 0.83217
Epoch 212/256
 - 6s - loss: 0.8941 - acc: 0.8509 - val_loss: 1.2753 - val_acc: 0.8252

Epoch 00212: val_acc did not improve from 0.83217
Epoch 213/256
 - 5s - loss: 0.9260 - acc: 0.8404 - val_loss: 1.5432 - val_acc: 0.6993

Epoch 00213: val_acc did not improve from 0.83217
Epoch 214/256
 - 6s - loss: 0.9394 - acc: 0.8175 - val_loss: 1.3042 - val_acc: 0.7413

Epoch 00214: val_acc did not improve from 0.83217
Epoch 215/256
 - 6s - loss: 0.8714 - acc: 0.8667 - val_loss: 2.2050 - val_acc: 0.5385

Epoch 00215: val_acc did not improve from 0.83217
Epoch 216/256
 - 6s - loss: 0.9736 - acc: 0.8316 - val_loss: 1.3142 - val_acc: 0.8042

Epoch 00216: val_acc did not improve from 0.83217
Epoch 217/256
 - 7s - loss: 0.9153 - acc: 0.8439 - val_loss: 2.2647 - val_acc: 0.6224

Epoch 00217: val_acc did not improve from 0.83217
Epoch 218/256
 - 6s - loss: 0.9338 - acc: 0.8421 - val_loss: 1.3836 - val_acc: 0.7692

Epoch 00218: val_acc did not improve from 0.83217
Epoch 219/256
 - 7s - loss: 0.9227 - acc: 0.8421 - val_loss: 1.2048 - val_acc: 0.7762

Epoch 00219: val_acc did not improve from 0.83217
Epoch 220/256
 - 6s - loss: 0.9400 - acc: 0.8351 - val_loss: 1.1761 - val_acc: 0.8042

Epoch 00220: val_acc did not improve from 0.83217
Epoch 221/256
 - 6s - loss: 0.9156 - acc: 0.8456 - val_loss: 1.2270 - val_acc: 0.7692

Epoch 00221: val_acc did not improve from 0.83217
Epoch 222/256
 - 6s - loss: 0.8795 - acc: 0.8579 - val_loss: 1.2428 - val_acc: 0.8042

Epoch 00222: val_acc did not improve from 0.83217
Epoch 223/256
 - 6s - loss: 0.8552 - acc: 0.8579 - val_loss: 1.2268 - val_acc: 0.8182

Epoch 00223: val_acc did not improve from 0.83217
Epoch 224/256
 - 6s - loss: 0.9415 - acc: 0.8351 - val_loss: 1.1005 - val_acc: 0.8252

Epoch 00224: val_acc did not improve from 0.83217
Epoch 225/256
 - 6s - loss: 0.9374 - acc: 0.8404 - val_loss: 1.1046 - val_acc: 0.8252

Epoch 00225: val_acc did not improve from 0.83217
Epoch 226/256
 - 5s - loss: 0.9186 - acc: 0.8456 - val_loss: 1.1840 - val_acc: 0.7902

Epoch 00226: val_acc did not improve from 0.83217
Epoch 227/256
 - 6s - loss: 0.8777 - acc: 0.8526 - val_loss: 1.0961 - val_acc: 0.8252

Epoch 00227: val_acc did not improve from 0.83217
Epoch 228/256
 - 6s - loss: 0.8899 - acc: 0.8333 - val_loss: 1.1356 - val_acc: 0.8182

Epoch 00228: val_acc did not improve from 0.83217
Epoch 229/256
 - 6s - loss: 1.0135 - acc: 0.8333 - val_loss: 1.4311 - val_acc: 0.7413

Epoch 00229: val_acc did not improve from 0.83217
Epoch 230/256
 - 6s - loss: 1.0033 - acc: 0.8053 - val_loss: 1.2632 - val_acc: 0.7972

Epoch 00230: val_acc did not improve from 0.83217
Epoch 231/256
 - 6s - loss: 0.9014 - acc: 0.8298 - val_loss: 1.3040 - val_acc: 0.7552

Epoch 00231: val_acc did not improve from 0.83217
Epoch 232/256
 - 6s - loss: 0.9421 - acc: 0.8316 - val_loss: 1.2325 - val_acc: 0.7832

Epoch 00232: val_acc did not improve from 0.83217
Epoch 233/256
 - 6s - loss: 0.9318 - acc: 0.8386 - val_loss: 1.1951 - val_acc: 0.8042

Epoch 00233: val_acc did not improve from 0.83217
Epoch 234/256
 - 6s - loss: 0.9643 - acc: 0.8298 - val_loss: 1.2487 - val_acc: 0.8042

Epoch 00234: val_acc did not improve from 0.83217
Epoch 235/256
 - 6s - loss: 0.8949 - acc: 0.8368 - val_loss: 1.3392 - val_acc: 0.8112

Epoch 00235: val_acc did not improve from 0.83217
Epoch 236/256
 - 6s - loss: 0.9198 - acc: 0.8596 - val_loss: 1.3399 - val_acc: 0.7902

Epoch 00236: val_acc did not improve from 0.83217
Epoch 237/256
 - 6s - loss: 0.8836 - acc: 0.8649 - val_loss: 1.1469 - val_acc: 0.8182

Epoch 00237: val_acc did not improve from 0.83217
Epoch 238/256
 - 7s - loss: 0.8499 - acc: 0.8702 - val_loss: 1.1855 - val_acc: 0.7762

Epoch 00238: val_acc did not improve from 0.83217
Epoch 239/256
 - 6s - loss: 0.9234 - acc: 0.8404 - val_loss: 1.1719 - val_acc: 0.8252

Epoch 00239: val_acc did not improve from 0.83217
Epoch 240/256
 - 6s - loss: 0.8586 - acc: 0.8456 - val_loss: 1.2108 - val_acc: 0.8112

Epoch 00240: val_acc did not improve from 0.83217
Epoch 241/256
 - 6s - loss: 0.8879 - acc: 0.8579 - val_loss: 1.5333 - val_acc: 0.7063

Epoch 00241: val_acc did not improve from 0.83217
Epoch 242/256
 - 6s - loss: 0.9310 - acc: 0.8421 - val_loss: 1.4025 - val_acc: 0.7762

Epoch 00242: val_acc did not improve from 0.83217
Epoch 243/256
 - 6s - loss: 0.8801 - acc: 0.8579 - val_loss: 1.2770 - val_acc: 0.8042

Epoch 00243: val_acc did not improve from 0.83217
Epoch 244/256
 - 6s - loss: 0.9118 - acc: 0.8404 - val_loss: 1.4015 - val_acc: 0.7063

Epoch 00244: val_acc did not improve from 0.83217
Epoch 245/256
 - 6s - loss: 0.9032 - acc: 0.8456 - val_loss: 1.2620 - val_acc: 0.8042

Epoch 00245: val_acc did not improve from 0.83217
Epoch 246/256
 - 7s - loss: 0.8413 - acc: 0.8667 - val_loss: 1.2189 - val_acc: 0.7762

Epoch 00246: val_acc did not improve from 0.83217
Epoch 247/256
 - 6s - loss: 0.8726 - acc: 0.8579 - val_loss: 1.3394 - val_acc: 0.7622

Epoch 00247: val_acc did not improve from 0.83217
Epoch 248/256
 - 6s - loss: 0.8601 - acc: 0.8579 - val_loss: 1.6103 - val_acc: 0.7692

Epoch 00248: val_acc did not improve from 0.83217
Epoch 249/256
 - 6s - loss: 0.8911 - acc: 0.8684 - val_loss: 1.7161 - val_acc: 0.6993

Epoch 00249: val_acc did not improve from 0.83217
Epoch 250/256
 - 6s - loss: 0.9098 - acc: 0.8491 - val_loss: 1.4240 - val_acc: 0.6923

Epoch 00250: val_acc did not improve from 0.83217
Epoch 251/256
 - 6s - loss: 0.8954 - acc: 0.8386 - val_loss: 1.3336 - val_acc: 0.7483

Epoch 00251: val_acc did not improve from 0.83217
Epoch 252/256
 - 6s - loss: 0.8862 - acc: 0.8561 - val_loss: 1.5236 - val_acc: 0.7622

Epoch 00252: val_acc did not improve from 0.83217
Epoch 253/256
 - 6s - loss: 0.9050 - acc: 0.8561 - val_loss: 1.2771 - val_acc: 0.8182

Epoch 00253: val_acc did not improve from 0.83217
Epoch 254/256
 - 5s - loss: 0.8372 - acc: 0.8632 - val_loss: 1.2514 - val_acc: 0.7832

Epoch 00254: val_acc did not improve from 0.83217
Epoch 255/256
 - 6s - loss: 0.8718 - acc: 0.8702 - val_loss: 1.2236 - val_acc: 0.8252

Epoch 00255: val_acc did not improve from 0.83217
Epoch 256/256
 - 6s - loss: 0.8619 - acc: 0.8632 - val_loss: 1.3566 - val_acc: 0.7413

Epoch 00256: val_acc did not improve from 0.83217
```
## Model Evaluation

### Class Dictionary

```
0 : Brush_teeth
1 : Climb_stairs
2 : Comb_hair
3 : Descend_stairs
4 : Drink_glass
5 : Eat_meat
6 : Eat_soup
7 : Getup_bed
8 : Liedown_bed
9 : Pour_water
10 : Sitdown_chair
11 : Standup_chair
12 : Use_telephone
13 : Walk
```
### Results

```
Test Loss: 0.9964
Test Accuracy: 0.8095

Confusion Matrix: 
[[ 1  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0 14  0  2  0  0  0  0  0  0  0  0  1]
 [ 0  0  1  0  0  0  1  0  0  0  0  0  0]
 [ 0  2  0  8  0  0  0  0  0  0  0  0  1]
 [ 0  0  0  0 13  0  0  1  2  0  0  1  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  8  0  0  0  2  0  0]
 [ 0  0  0  0  1  0  0  4  0  2  0  0  0]
 [ 0  0  0  0  3  0  0  0 12  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0 13  0  0  0]
 [ 0  1  0  0  0  0  3  0  0  0 16  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  1  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0 11]]

Classification Report: 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         1
           1       0.82      0.82      0.82        17
           2       1.00      0.50      0.67         2
           3       0.80      0.73      0.76        11
           4       0.76      0.76      0.76        17
           5       0.00      0.00      0.00         1
           6       0.00      0.00      0.00         0
           7       0.67      0.80      0.73        10
           8       0.80      0.57      0.67         7
           9       0.86      0.80      0.83        15
          10       0.87      1.00      0.93        13
          11       0.89      0.80      0.84        20
          12       0.50      1.00      0.67         1
          13       0.85      1.00      0.92        11

    accuracy                           0.81       126
   macro avg       0.70      0.70      0.69       126
weighted avg       0.82      0.81      0.81       126

```
