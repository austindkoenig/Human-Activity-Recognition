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
 - 46s - loss: 65.6112 - acc: 0.1105 - val_loss: 63.7463 - val_acc: 0.0979

Epoch 00001: val_acc improved from -inf to 0.09790, saving model to ./output/model.hdf5
Epoch 2/256
 - 7s - loss: 62.0012 - acc: 0.1298 - val_loss: 60.2351 - val_acc: 0.1049

Epoch 00002: val_acc improved from 0.09790 to 0.10490, saving model to ./output/model.hdf5
Epoch 3/256
 - 7s - loss: 58.6255 - acc: 0.1105 - val_loss: 56.9626 - val_acc: 0.1049

Epoch 00003: val_acc did not improve from 0.10490
Epoch 4/256
 - 6s - loss: 55.4868 - acc: 0.1246 - val_loss: 53.9331 - val_acc: 0.1958

Epoch 00004: val_acc improved from 0.10490 to 0.19580, saving model to ./output/model.hdf5
Epoch 5/256
 - 6s - loss: 52.5198 - acc: 0.1579 - val_loss: 51.0035 - val_acc: 0.2098

Epoch 00005: val_acc improved from 0.19580 to 0.20979, saving model to ./output/model.hdf5
Epoch 6/256
 - 6s - loss: 49.6591 - acc: 0.2509 - val_loss: 48.1530 - val_acc: 0.3497

Epoch 00006: val_acc improved from 0.20979 to 0.34965, saving model to ./output/model.hdf5
Epoch 7/256
 - 5s - loss: 46.7251 - acc: 0.3614 - val_loss: 45.2712 - val_acc: 0.3776

Epoch 00007: val_acc improved from 0.34965 to 0.37762, saving model to ./output/model.hdf5
Epoch 8/256
 - 6s - loss: 43.8966 - acc: 0.4842 - val_loss: 42.2924 - val_acc: 0.5804

Epoch 00008: val_acc improved from 0.37762 to 0.58042, saving model to ./output/model.hdf5
Epoch 9/256
 - 5s - loss: 41.3480 - acc: 0.5684 - val_loss: 40.0870 - val_acc: 0.6294

Epoch 00009: val_acc improved from 0.58042 to 0.62937, saving model to ./output/model.hdf5
Epoch 10/256
 - 6s - loss: 39.0585 - acc: 0.5877 - val_loss: 37.7279 - val_acc: 0.6643

Epoch 00010: val_acc improved from 0.62937 to 0.66434, saving model to ./output/model.hdf5
Epoch 11/256
 - 6s - loss: 36.8933 - acc: 0.6035 - val_loss: 35.8439 - val_acc: 0.6014

Epoch 00011: val_acc did not improve from 0.66434
Epoch 12/256
 - 6s - loss: 34.8474 - acc: 0.6386 - val_loss: 33.7636 - val_acc: 0.6643

Epoch 00012: val_acc did not improve from 0.66434
Epoch 13/256
 - 6s - loss: 32.9179 - acc: 0.6737 - val_loss: 32.1479 - val_acc: 0.6713

Epoch 00013: val_acc improved from 0.66434 to 0.67133, saving model to ./output/model.hdf5
Epoch 14/256
 - 6s - loss: 31.1394 - acc: 0.6544 - val_loss: 30.1258 - val_acc: 0.7203

Epoch 00014: val_acc improved from 0.67133 to 0.72028, saving model to ./output/model.hdf5
Epoch 15/256
 - 6s - loss: 29.4389 - acc: 0.6667 - val_loss: 28.4208 - val_acc: 0.7483

Epoch 00015: val_acc improved from 0.72028 to 0.74825, saving model to ./output/model.hdf5
Epoch 16/256
 - 5s - loss: 27.8933 - acc: 0.6561 - val_loss: 27.0058 - val_acc: 0.7552

Epoch 00016: val_acc improved from 0.74825 to 0.75524, saving model to ./output/model.hdf5
Epoch 17/256
 - 5s - loss: 26.3746 - acc: 0.6877 - val_loss: 25.6137 - val_acc: 0.6923

Epoch 00017: val_acc did not improve from 0.75524
Epoch 18/256
 - 5s - loss: 24.9635 - acc: 0.6684 - val_loss: 24.2689 - val_acc: 0.6923

Epoch 00018: val_acc did not improve from 0.75524
Epoch 19/256
 - 5s - loss: 23.6334 - acc: 0.6930 - val_loss: 22.8484 - val_acc: 0.7832

Epoch 00019: val_acc improved from 0.75524 to 0.78322, saving model to ./output/model.hdf5
Epoch 20/256
 - 6s - loss: 22.3281 - acc: 0.7088 - val_loss: 21.6069 - val_acc: 0.7692

Epoch 00020: val_acc did not improve from 0.78322
Epoch 21/256
 - 6s - loss: 21.2131 - acc: 0.7070 - val_loss: 21.0515 - val_acc: 0.5664

Epoch 00021: val_acc did not improve from 0.78322
Epoch 22/256
 - 5s - loss: 20.0361 - acc: 0.7088 - val_loss: 19.4918 - val_acc: 0.7413

Epoch 00022: val_acc did not improve from 0.78322
Epoch 23/256
 - 5s - loss: 19.0269 - acc: 0.6930 - val_loss: 18.4932 - val_acc: 0.6853

Epoch 00023: val_acc did not improve from 0.78322
Epoch 24/256
 - 6s - loss: 18.0216 - acc: 0.7123 - val_loss: 17.5242 - val_acc: 0.6853

Epoch 00024: val_acc did not improve from 0.78322
Epoch 25/256
 - 6s - loss: 17.0302 - acc: 0.7140 - val_loss: 16.5847 - val_acc: 0.7692

Epoch 00025: val_acc did not improve from 0.78322
Epoch 26/256
 - 5s - loss: 16.1869 - acc: 0.6930 - val_loss: 15.6340 - val_acc: 0.7972

Epoch 00026: val_acc improved from 0.78322 to 0.79720, saving model to ./output/model.hdf5
Epoch 27/256
 - 6s - loss: 15.3171 - acc: 0.7456 - val_loss: 14.7771 - val_acc: 0.7972

Epoch 00027: val_acc did not improve from 0.79720
Epoch 28/256
 - 5s - loss: 14.5456 - acc: 0.7088 - val_loss: 14.1333 - val_acc: 0.7273

Epoch 00028: val_acc did not improve from 0.79720
Epoch 29/256
 - 6s - loss: 13.7532 - acc: 0.7211 - val_loss: 13.3919 - val_acc: 0.7203

Epoch 00029: val_acc did not improve from 0.79720
Epoch 30/256
 - 6s - loss: 13.0508 - acc: 0.7333 - val_loss: 12.6729 - val_acc: 0.7343

Epoch 00030: val_acc did not improve from 0.79720
Epoch 31/256
 - 5s - loss: 12.3603 - acc: 0.7333 - val_loss: 11.9624 - val_acc: 0.7902

Epoch 00031: val_acc did not improve from 0.79720
Epoch 32/256
 - 6s - loss: 11.7430 - acc: 0.7263 - val_loss: 11.3793 - val_acc: 0.7413

Epoch 00032: val_acc did not improve from 0.79720
Epoch 33/256
 - 6s - loss: 11.1500 - acc: 0.7281 - val_loss: 10.8966 - val_acc: 0.7343

Epoch 00033: val_acc did not improve from 0.79720
Epoch 34/256
 - 5s - loss: 10.6140 - acc: 0.7158 - val_loss: 10.3430 - val_acc: 0.7413

Epoch 00034: val_acc did not improve from 0.79720
Epoch 35/256
 - 5s - loss: 10.0664 - acc: 0.7421 - val_loss: 9.7567 - val_acc: 0.7343

Epoch 00035: val_acc did not improve from 0.79720
Epoch 36/256
 - 6s - loss: 9.5611 - acc: 0.7386 - val_loss: 9.2680 - val_acc: 0.7413

Epoch 00036: val_acc did not improve from 0.79720
Epoch 37/256
 - 6s - loss: 9.0880 - acc: 0.7246 - val_loss: 8.8025 - val_acc: 0.7483

Epoch 00037: val_acc did not improve from 0.79720
Epoch 38/256
 - 6s - loss: 8.6333 - acc: 0.7298 - val_loss: 8.4091 - val_acc: 0.7343

Epoch 00038: val_acc did not improve from 0.79720
Epoch 39/256
 - 6s - loss: 8.1993 - acc: 0.7368 - val_loss: 8.0087 - val_acc: 0.7273

Epoch 00039: val_acc did not improve from 0.79720
Epoch 40/256
 - 6s - loss: 7.8232 - acc: 0.7316 - val_loss: 7.7267 - val_acc: 0.7552

Epoch 00040: val_acc did not improve from 0.79720
Epoch 41/256
 - 6s - loss: 7.4264 - acc: 0.7316 - val_loss: 7.3610 - val_acc: 0.7203

Epoch 00041: val_acc did not improve from 0.79720
Epoch 42/256
 - 6s - loss: 7.0729 - acc: 0.7404 - val_loss: 6.8746 - val_acc: 0.7622

Epoch 00042: val_acc did not improve from 0.79720
Epoch 43/256
 - 6s - loss: 6.7608 - acc: 0.7246 - val_loss: 6.5236 - val_acc: 0.7483

Epoch 00043: val_acc did not improve from 0.79720
Epoch 44/256
 - 6s - loss: 6.3619 - acc: 0.7509 - val_loss: 6.3425 - val_acc: 0.7203

Epoch 00044: val_acc did not improve from 0.79720
Epoch 45/256
 - 6s - loss: 6.0889 - acc: 0.7579 - val_loss: 6.0196 - val_acc: 0.7552

Epoch 00045: val_acc did not improve from 0.79720
Epoch 46/256
 - 6s - loss: 5.8154 - acc: 0.7737 - val_loss: 5.7331 - val_acc: 0.7552

Epoch 00046: val_acc did not improve from 0.79720
Epoch 47/256
 - 7s - loss: 5.5729 - acc: 0.7596 - val_loss: 5.4725 - val_acc: 0.7552

Epoch 00047: val_acc did not improve from 0.79720
Epoch 48/256
 - 5s - loss: 5.3184 - acc: 0.7544 - val_loss: 5.9224 - val_acc: 0.5804

Epoch 00048: val_acc did not improve from 0.79720
Epoch 49/256
 - 6s - loss: 5.0746 - acc: 0.7474 - val_loss: 5.1595 - val_acc: 0.6993

Epoch 00049: val_acc did not improve from 0.79720
Epoch 50/256
 - 6s - loss: 4.8231 - acc: 0.7754 - val_loss: 4.8394 - val_acc: 0.7483

Epoch 00050: val_acc did not improve from 0.79720
Epoch 51/256
 - 6s - loss: 4.6265 - acc: 0.7596 - val_loss: 4.5497 - val_acc: 0.7832

Epoch 00051: val_acc did not improve from 0.79720
Epoch 52/256
 - 6s - loss: 4.4347 - acc: 0.7474 - val_loss: 4.6285 - val_acc: 0.6573

Epoch 00052: val_acc did not improve from 0.79720
Epoch 53/256
 - 6s - loss: 4.1904 - acc: 0.7719 - val_loss: 4.2083 - val_acc: 0.7762

Epoch 00053: val_acc did not improve from 0.79720
Epoch 54/256
 - 6s - loss: 4.0623 - acc: 0.7895 - val_loss: 3.9892 - val_acc: 0.7762

Epoch 00054: val_acc did not improve from 0.79720
Epoch 55/256
 - 7s - loss: 3.8748 - acc: 0.7719 - val_loss: 3.9262 - val_acc: 0.7413

Epoch 00055: val_acc did not improve from 0.79720
Epoch 56/256
 - 7s - loss: 3.7005 - acc: 0.7772 - val_loss: 3.7230 - val_acc: 0.7972

Epoch 00056: val_acc did not improve from 0.79720
Epoch 57/256
 - 6s - loss: 3.5514 - acc: 0.7474 - val_loss: 3.5352 - val_acc: 0.7622

Epoch 00057: val_acc did not improve from 0.79720
Epoch 58/256
 - 5s - loss: 3.4006 - acc: 0.7649 - val_loss: 3.3744 - val_acc: 0.7832

Epoch 00058: val_acc did not improve from 0.79720
Epoch 59/256
 - 6s - loss: 3.2589 - acc: 0.7561 - val_loss: 3.4400 - val_acc: 0.7273

Epoch 00059: val_acc did not improve from 0.79720
Epoch 60/256
 - 6s - loss: 3.1145 - acc: 0.7965 - val_loss: 3.3533 - val_acc: 0.7063

Epoch 00060: val_acc did not improve from 0.79720
Epoch 61/256
 - 7s - loss: 2.9666 - acc: 0.8018 - val_loss: 3.1683 - val_acc: 0.7343

Epoch 00061: val_acc did not improve from 0.79720
Epoch 62/256
 - 6s - loss: 2.9379 - acc: 0.7439 - val_loss: 3.9667 - val_acc: 0.5664

Epoch 00062: val_acc did not improve from 0.79720
Epoch 63/256
 - 6s - loss: 2.8535 - acc: 0.7614 - val_loss: 2.9182 - val_acc: 0.7552

Epoch 00063: val_acc did not improve from 0.79720
Epoch 64/256
 - 6s - loss: 2.7240 - acc: 0.7632 - val_loss: 2.6915 - val_acc: 0.7622

Epoch 00064: val_acc did not improve from 0.79720
Epoch 65/256
 - 5s - loss: 2.6045 - acc: 0.7702 - val_loss: 2.9945 - val_acc: 0.6573

Epoch 00065: val_acc did not improve from 0.79720
Epoch 66/256
 - 6s - loss: 2.5053 - acc: 0.7825 - val_loss: 2.6156 - val_acc: 0.7762

Epoch 00066: val_acc did not improve from 0.79720
Epoch 67/256
 - 6s - loss: 2.4363 - acc: 0.7877 - val_loss: 2.4594 - val_acc: 0.7552

Epoch 00067: val_acc did not improve from 0.79720
Epoch 68/256
 - 7s - loss: 2.3610 - acc: 0.7825 - val_loss: 2.3739 - val_acc: 0.7692

Epoch 00068: val_acc did not improve from 0.79720
Epoch 69/256
 - 6s - loss: 2.2755 - acc: 0.7754 - val_loss: 2.3283 - val_acc: 0.7692

Epoch 00069: val_acc did not improve from 0.79720
Epoch 70/256
 - 6s - loss: 2.2237 - acc: 0.7754 - val_loss: 2.3035 - val_acc: 0.7483

Epoch 00070: val_acc did not improve from 0.79720
Epoch 71/256
 - 6s - loss: 2.1503 - acc: 0.7772 - val_loss: 2.2537 - val_acc: 0.7483

Epoch 00071: val_acc did not improve from 0.79720
Epoch 72/256
 - 6s - loss: 2.0697 - acc: 0.7754 - val_loss: 2.2168 - val_acc: 0.7483

Epoch 00072: val_acc did not improve from 0.79720
Epoch 73/256
 - 6s - loss: 2.0065 - acc: 0.7772 - val_loss: 2.0613 - val_acc: 0.8042

Epoch 00073: val_acc improved from 0.79720 to 0.80420, saving model to ./output/model.hdf5
Epoch 74/256
 - 6s - loss: 1.9876 - acc: 0.7667 - val_loss: 2.1278 - val_acc: 0.7692

Epoch 00074: val_acc did not improve from 0.80420
Epoch 75/256
 - 6s - loss: 1.8998 - acc: 0.7649 - val_loss: 2.2928 - val_acc: 0.6853

Epoch 00075: val_acc did not improve from 0.80420
Epoch 76/256
 - 6s - loss: 1.8393 - acc: 0.7930 - val_loss: 1.9654 - val_acc: 0.7692

Epoch 00076: val_acc did not improve from 0.80420
Epoch 77/256
 - 6s - loss: 1.7653 - acc: 0.8088 - val_loss: 1.9026 - val_acc: 0.7902

Epoch 00077: val_acc did not improve from 0.80420
Epoch 78/256
 - 6s - loss: 1.7570 - acc: 0.7860 - val_loss: 1.8442 - val_acc: 0.7902

Epoch 00078: val_acc did not improve from 0.80420
Epoch 79/256
 - 6s - loss: 1.7260 - acc: 0.7860 - val_loss: 1.9595 - val_acc: 0.7692

Epoch 00079: val_acc did not improve from 0.80420
Epoch 80/256
 - 6s - loss: 1.6591 - acc: 0.7842 - val_loss: 1.8082 - val_acc: 0.7552

Epoch 00080: val_acc did not improve from 0.80420
Epoch 81/256
 - 7s - loss: 1.6147 - acc: 0.7702 - val_loss: 1.5843 - val_acc: 0.8042

Epoch 00081: val_acc did not improve from 0.80420
Epoch 82/256
 - 7s - loss: 1.5600 - acc: 0.8123 - val_loss: 1.6884 - val_acc: 0.7692

Epoch 00082: val_acc did not improve from 0.80420
Epoch 83/256
 - 5s - loss: 1.5727 - acc: 0.7912 - val_loss: 1.7315 - val_acc: 0.7343

Epoch 00083: val_acc did not improve from 0.80420
Epoch 84/256
 - 7s - loss: 1.5469 - acc: 0.7772 - val_loss: 1.5828 - val_acc: 0.7762

Epoch 00084: val_acc did not improve from 0.80420
Epoch 85/256
 - 6s - loss: 1.4813 - acc: 0.7649 - val_loss: 1.7167 - val_acc: 0.7343

Epoch 00085: val_acc did not improve from 0.80420
Epoch 86/256
 - 6s - loss: 1.4832 - acc: 0.7807 - val_loss: 1.6991 - val_acc: 0.7622

Epoch 00086: val_acc did not improve from 0.80420
Epoch 87/256
 - 7s - loss: 1.4179 - acc: 0.7965 - val_loss: 1.5687 - val_acc: 0.7762

Epoch 00087: val_acc did not improve from 0.80420
Epoch 88/256
 - 7s - loss: 1.3733 - acc: 0.7947 - val_loss: 1.8181 - val_acc: 0.6993

Epoch 00088: val_acc did not improve from 0.80420
Epoch 89/256
 - 7s - loss: 1.3779 - acc: 0.7825 - val_loss: 1.5442 - val_acc: 0.7762

Epoch 00089: val_acc did not improve from 0.80420
Epoch 90/256
 - 6s - loss: 1.3574 - acc: 0.7789 - val_loss: 1.5333 - val_acc: 0.7762

Epoch 00090: val_acc did not improve from 0.80420
Epoch 91/256
 - 6s - loss: 1.3385 - acc: 0.7912 - val_loss: 1.4874 - val_acc: 0.7552

Epoch 00091: val_acc did not improve from 0.80420
Epoch 92/256
 - 6s - loss: 1.2830 - acc: 0.7930 - val_loss: 1.5003 - val_acc: 0.7692

Epoch 00092: val_acc did not improve from 0.80420
Epoch 93/256
 - 6s - loss: 1.2770 - acc: 0.7930 - val_loss: 1.4879 - val_acc: 0.7413

Epoch 00093: val_acc did not improve from 0.80420
Epoch 94/256
 - 6s - loss: 1.3052 - acc: 0.7772 - val_loss: 1.4334 - val_acc: 0.7483

Epoch 00094: val_acc did not improve from 0.80420
Epoch 95/256
 - 6s - loss: 1.2320 - acc: 0.8000 - val_loss: 1.4433 - val_acc: 0.7902

Epoch 00095: val_acc did not improve from 0.80420
Epoch 96/256
 - 6s - loss: 1.2353 - acc: 0.8123 - val_loss: 1.4307 - val_acc: 0.7762

Epoch 00096: val_acc did not improve from 0.80420
Epoch 97/256
 - 6s - loss: 1.2115 - acc: 0.8175 - val_loss: 1.3909 - val_acc: 0.7692

Epoch 00097: val_acc did not improve from 0.80420
Epoch 98/256
 - 6s - loss: 1.2255 - acc: 0.7842 - val_loss: 1.4657 - val_acc: 0.7343

Epoch 00098: val_acc did not improve from 0.80420
Epoch 99/256
 - 6s - loss: 1.1538 - acc: 0.8193 - val_loss: 2.1673 - val_acc: 0.6434

Epoch 00099: val_acc did not improve from 0.80420
Epoch 100/256
 - 6s - loss: 1.1546 - acc: 0.7982 - val_loss: 1.3487 - val_acc: 0.7762

Epoch 00100: val_acc did not improve from 0.80420
Epoch 101/256
 - 6s - loss: 1.1135 - acc: 0.8088 - val_loss: 1.3746 - val_acc: 0.7622

Epoch 00101: val_acc did not improve from 0.80420
Epoch 102/256
 - 6s - loss: 1.1601 - acc: 0.7877 - val_loss: 1.6860 - val_acc: 0.6643

Epoch 00102: val_acc did not improve from 0.80420
Epoch 103/256
 - 5s - loss: 1.1655 - acc: 0.7947 - val_loss: 1.3913 - val_acc: 0.7552

Epoch 00103: val_acc did not improve from 0.80420
Epoch 104/256
 - 6s - loss: 1.1797 - acc: 0.7737 - val_loss: 1.3097 - val_acc: 0.7483

Epoch 00104: val_acc did not improve from 0.80420
Epoch 105/256
 - 6s - loss: 1.1152 - acc: 0.8018 - val_loss: 1.4008 - val_acc: 0.7762

Epoch 00105: val_acc did not improve from 0.80420
Epoch 106/256
 - 6s - loss: 1.1158 - acc: 0.8053 - val_loss: 1.4703 - val_acc: 0.6993

Epoch 00106: val_acc did not improve from 0.80420
Epoch 107/256
 - 5s - loss: 1.1210 - acc: 0.8070 - val_loss: 1.3131 - val_acc: 0.7622

Epoch 00107: val_acc did not improve from 0.80420
Epoch 108/256
 - 6s - loss: 1.0701 - acc: 0.8175 - val_loss: 1.2761 - val_acc: 0.7552

Epoch 00108: val_acc did not improve from 0.80420
Epoch 109/256
 - 6s - loss: 1.0410 - acc: 0.8228 - val_loss: 1.2767 - val_acc: 0.7622

Epoch 00109: val_acc did not improve from 0.80420
Epoch 110/256
 - 6s - loss: 1.0832 - acc: 0.8000 - val_loss: 1.2644 - val_acc: 0.7552

Epoch 00110: val_acc did not improve from 0.80420
Epoch 111/256
 - 6s - loss: 1.0717 - acc: 0.8193 - val_loss: 1.4432 - val_acc: 0.6853

Epoch 00111: val_acc did not improve from 0.80420
Epoch 112/256
 - 7s - loss: 1.1033 - acc: 0.7912 - val_loss: 1.3399 - val_acc: 0.7413

Epoch 00112: val_acc did not improve from 0.80420
Epoch 113/256
 - 6s - loss: 1.0579 - acc: 0.8088 - val_loss: 1.1451 - val_acc: 0.7832

Epoch 00113: val_acc did not improve from 0.80420
Epoch 114/256
 - 6s - loss: 1.0392 - acc: 0.8070 - val_loss: 1.8120 - val_acc: 0.6364

Epoch 00114: val_acc did not improve from 0.80420
Epoch 115/256
 - 7s - loss: 1.0854 - acc: 0.7965 - val_loss: 1.2041 - val_acc: 0.7902

Epoch 00115: val_acc did not improve from 0.80420
Epoch 116/256
 - 6s - loss: 1.0414 - acc: 0.7965 - val_loss: 1.3653 - val_acc: 0.7483

Epoch 00116: val_acc did not improve from 0.80420
Epoch 117/256
 - 6s - loss: 1.0760 - acc: 0.7877 - val_loss: 1.2330 - val_acc: 0.7552

Epoch 00117: val_acc did not improve from 0.80420
Epoch 118/256
 - 6s - loss: 1.0468 - acc: 0.7912 - val_loss: 1.2216 - val_acc: 0.7972

Epoch 00118: val_acc did not improve from 0.80420
Epoch 119/256
 - 6s - loss: 1.0425 - acc: 0.7982 - val_loss: 1.4414 - val_acc: 0.6783

Epoch 00119: val_acc did not improve from 0.80420
Epoch 120/256
 - 6s - loss: 1.0122 - acc: 0.7877 - val_loss: 1.2865 - val_acc: 0.7483

Epoch 00120: val_acc did not improve from 0.80420
Epoch 121/256
 - 6s - loss: 1.0005 - acc: 0.8123 - val_loss: 1.4655 - val_acc: 0.7133

Epoch 00121: val_acc did not improve from 0.80420
Epoch 122/256
 - 6s - loss: 1.0645 - acc: 0.7842 - val_loss: 1.1123 - val_acc: 0.7972

Epoch 00122: val_acc did not improve from 0.80420
Epoch 123/256
 - 6s - loss: 0.9769 - acc: 0.8105 - val_loss: 1.2826 - val_acc: 0.7413

Epoch 00123: val_acc did not improve from 0.80420
Epoch 124/256
 - 5s - loss: 1.0492 - acc: 0.7877 - val_loss: 1.2592 - val_acc: 0.7762

Epoch 00124: val_acc did not improve from 0.80420
Epoch 125/256
 - 6s - loss: 1.0393 - acc: 0.7842 - val_loss: 1.3111 - val_acc: 0.7552

Epoch 00125: val_acc did not improve from 0.80420
Epoch 126/256
 - 6s - loss: 0.9841 - acc: 0.8246 - val_loss: 1.2962 - val_acc: 0.7972

Epoch 00126: val_acc did not improve from 0.80420
Epoch 127/256
 - 6s - loss: 1.0215 - acc: 0.8053 - val_loss: 1.0915 - val_acc: 0.8182

Epoch 00127: val_acc improved from 0.80420 to 0.81818, saving model to ./output/model.hdf5
Epoch 128/256
 - 7s - loss: 0.9787 - acc: 0.8018 - val_loss: 1.3019 - val_acc: 0.7622

Epoch 00128: val_acc did not improve from 0.81818
Epoch 129/256
 - 5s - loss: 0.9268 - acc: 0.8333 - val_loss: 1.2779 - val_acc: 0.7692

Epoch 00129: val_acc did not improve from 0.81818
Epoch 130/256
 - 5s - loss: 0.9731 - acc: 0.8035 - val_loss: 1.1335 - val_acc: 0.7622

Epoch 00130: val_acc did not improve from 0.81818
Epoch 131/256
 - 6s - loss: 0.9731 - acc: 0.8228 - val_loss: 1.1267 - val_acc: 0.7902

Epoch 00131: val_acc did not improve from 0.81818
Epoch 132/256
 - 5s - loss: 0.9722 - acc: 0.8123 - val_loss: 1.2641 - val_acc: 0.7552

Epoch 00132: val_acc did not improve from 0.81818
Epoch 133/256
 - 6s - loss: 1.0096 - acc: 0.7912 - val_loss: 1.4589 - val_acc: 0.7343

Epoch 00133: val_acc did not improve from 0.81818
Epoch 134/256
 - 6s - loss: 0.9476 - acc: 0.7965 - val_loss: 1.1538 - val_acc: 0.7762

Epoch 00134: val_acc did not improve from 0.81818
Epoch 135/256
 - 6s - loss: 0.9566 - acc: 0.8263 - val_loss: 1.2801 - val_acc: 0.7692

Epoch 00135: val_acc did not improve from 0.81818
Epoch 136/256
 - 7s - loss: 0.9534 - acc: 0.8175 - val_loss: 1.1086 - val_acc: 0.7832

Epoch 00136: val_acc did not improve from 0.81818
Epoch 137/256
 - 6s - loss: 0.9644 - acc: 0.8175 - val_loss: 1.2887 - val_acc: 0.7413

Epoch 00137: val_acc did not improve from 0.81818
Epoch 138/256
 - 5s - loss: 0.9945 - acc: 0.8000 - val_loss: 1.3868 - val_acc: 0.7203

Epoch 00138: val_acc did not improve from 0.81818
Epoch 139/256
 - 7s - loss: 0.9361 - acc: 0.8158 - val_loss: 1.2918 - val_acc: 0.7552

Epoch 00139: val_acc did not improve from 0.81818
Epoch 140/256
 - 6s - loss: 0.9125 - acc: 0.8228 - val_loss: 1.1141 - val_acc: 0.7902

Epoch 00140: val_acc did not improve from 0.81818
Epoch 141/256
 - 5s - loss: 0.9005 - acc: 0.8246 - val_loss: 1.2591 - val_acc: 0.7692

Epoch 00141: val_acc did not improve from 0.81818
Epoch 142/256
 - 6s - loss: 0.9632 - acc: 0.8088 - val_loss: 1.3670 - val_acc: 0.6923

Epoch 00142: val_acc did not improve from 0.81818
Epoch 143/256
 - 6s - loss: 0.9368 - acc: 0.8018 - val_loss: 1.3540 - val_acc: 0.7552

Epoch 00143: val_acc did not improve from 0.81818
Epoch 144/256
 - 6s - loss: 0.9468 - acc: 0.8140 - val_loss: 1.3559 - val_acc: 0.8182

Epoch 00144: val_acc did not improve from 0.81818
Epoch 145/256
 - 6s - loss: 0.9805 - acc: 0.7912 - val_loss: 1.2281 - val_acc: 0.7692

Epoch 00145: val_acc did not improve from 0.81818
Epoch 146/256
 - 6s - loss: 0.9524 - acc: 0.8105 - val_loss: 1.3694 - val_acc: 0.7343

Epoch 00146: val_acc did not improve from 0.81818
Epoch 147/256
 - 6s - loss: 0.9781 - acc: 0.8246 - val_loss: 1.2361 - val_acc: 0.7552

Epoch 00147: val_acc did not improve from 0.81818
Epoch 148/256
 - 6s - loss: 0.9133 - acc: 0.8246 - val_loss: 2.1262 - val_acc: 0.5664

Epoch 00148: val_acc did not improve from 0.81818
Epoch 149/256
 - 6s - loss: 0.9423 - acc: 0.8263 - val_loss: 1.2685 - val_acc: 0.7902

Epoch 00149: val_acc did not improve from 0.81818
Epoch 150/256
 - 6s - loss: 0.9518 - acc: 0.8175 - val_loss: 1.4034 - val_acc: 0.7273

Epoch 00150: val_acc did not improve from 0.81818
Epoch 151/256
 - 6s - loss: 0.9802 - acc: 0.8123 - val_loss: 1.1439 - val_acc: 0.7972

Epoch 00151: val_acc did not improve from 0.81818
Epoch 152/256
 - 7s - loss: 0.9770 - acc: 0.8018 - val_loss: 1.1301 - val_acc: 0.7972

Epoch 00152: val_acc did not improve from 0.81818
Epoch 153/256
 - 6s - loss: 0.9354 - acc: 0.8281 - val_loss: 1.3166 - val_acc: 0.7483

Epoch 00153: val_acc did not improve from 0.81818
Epoch 154/256
 - 6s - loss: 0.9438 - acc: 0.8246 - val_loss: 1.1551 - val_acc: 0.7762

Epoch 00154: val_acc did not improve from 0.81818
Epoch 155/256
 - 6s - loss: 0.9299 - acc: 0.8281 - val_loss: 1.1804 - val_acc: 0.7692

Epoch 00155: val_acc did not improve from 0.81818
Epoch 156/256
 - 6s - loss: 0.9617 - acc: 0.8281 - val_loss: 1.2555 - val_acc: 0.7972

Epoch 00156: val_acc did not improve from 0.81818
Epoch 157/256
 - 6s - loss: 0.9123 - acc: 0.8333 - val_loss: 1.2104 - val_acc: 0.7413

Epoch 00157: val_acc did not improve from 0.81818
Epoch 158/256
 - 7s - loss: 0.9769 - acc: 0.8123 - val_loss: 1.1271 - val_acc: 0.7762

Epoch 00158: val_acc did not improve from 0.81818
Epoch 159/256
 - 6s - loss: 0.9170 - acc: 0.8333 - val_loss: 1.1858 - val_acc: 0.7902

Epoch 00159: val_acc did not improve from 0.81818
Epoch 160/256
 - 6s - loss: 0.9690 - acc: 0.8123 - val_loss: 1.5233 - val_acc: 0.6923

Epoch 00160: val_acc did not improve from 0.81818
Epoch 161/256
 - 7s - loss: 0.9185 - acc: 0.8228 - val_loss: 1.2870 - val_acc: 0.7762

Epoch 00161: val_acc did not improve from 0.81818
Epoch 162/256
 - 7s - loss: 0.9274 - acc: 0.8193 - val_loss: 1.1872 - val_acc: 0.8462

Epoch 00162: val_acc improved from 0.81818 to 0.84615, saving model to ./output/model.hdf5
Epoch 163/256
 - 6s - loss: 0.9629 - acc: 0.7965 - val_loss: 2.2461 - val_acc: 0.5594

Epoch 00163: val_acc did not improve from 0.84615
Epoch 164/256
 - 6s - loss: 0.9465 - acc: 0.8263 - val_loss: 1.2378 - val_acc: 0.7972

Epoch 00164: val_acc did not improve from 0.84615
Epoch 165/256
 - 5s - loss: 0.9134 - acc: 0.8246 - val_loss: 1.3448 - val_acc: 0.7552

Epoch 00165: val_acc did not improve from 0.84615
Epoch 166/256
 - 6s - loss: 0.9165 - acc: 0.8211 - val_loss: 1.3680 - val_acc: 0.7762

Epoch 00166: val_acc did not improve from 0.84615
Epoch 167/256
 - 6s - loss: 0.9476 - acc: 0.8281 - val_loss: 1.3441 - val_acc: 0.7552

Epoch 00167: val_acc did not improve from 0.84615
Epoch 168/256
 - 7s - loss: 0.9298 - acc: 0.8263 - val_loss: 1.2941 - val_acc: 0.7902

Epoch 00168: val_acc did not improve from 0.84615
Epoch 169/256
 - 6s - loss: 0.9972 - acc: 0.8105 - val_loss: 1.1292 - val_acc: 0.8042

Epoch 00169: val_acc did not improve from 0.84615
Epoch 170/256
 - 6s - loss: 0.8726 - acc: 0.8456 - val_loss: 1.3156 - val_acc: 0.7762

Epoch 00170: val_acc did not improve from 0.84615
Epoch 171/256
 - 6s - loss: 0.9923 - acc: 0.8035 - val_loss: 1.3198 - val_acc: 0.7902

Epoch 00171: val_acc did not improve from 0.84615
Epoch 172/256
 - 6s - loss: 0.9237 - acc: 0.8351 - val_loss: 1.6163 - val_acc: 0.7483

Epoch 00172: val_acc did not improve from 0.84615
Epoch 173/256
 - 6s - loss: 0.8813 - acc: 0.8649 - val_loss: 1.2122 - val_acc: 0.7902

Epoch 00173: val_acc did not improve from 0.84615
Epoch 174/256
 - 6s - loss: 0.8994 - acc: 0.8333 - val_loss: 1.0886 - val_acc: 0.8182

Epoch 00174: val_acc did not improve from 0.84615
Epoch 175/256
 - 6s - loss: 0.9465 - acc: 0.8333 - val_loss: 1.2603 - val_acc: 0.7552

Epoch 00175: val_acc did not improve from 0.84615
Epoch 176/256
 - 6s - loss: 0.9448 - acc: 0.8246 - val_loss: 1.2768 - val_acc: 0.7343

Epoch 00176: val_acc did not improve from 0.84615
Epoch 177/256
 - 7s - loss: 0.9629 - acc: 0.8158 - val_loss: 1.2914 - val_acc: 0.7343

Epoch 00177: val_acc did not improve from 0.84615
Epoch 178/256
 - 6s - loss: 0.9238 - acc: 0.8439 - val_loss: 1.3243 - val_acc: 0.7552

Epoch 00178: val_acc did not improve from 0.84615
Epoch 179/256
 - 6s - loss: 0.9251 - acc: 0.8404 - val_loss: 1.4420 - val_acc: 0.7483

Epoch 00179: val_acc did not improve from 0.84615
Epoch 180/256
 - 6s - loss: 0.9233 - acc: 0.8368 - val_loss: 1.4325 - val_acc: 0.7343

Epoch 00180: val_acc did not improve from 0.84615
Epoch 181/256
 - 6s - loss: 0.9029 - acc: 0.8421 - val_loss: 1.3980 - val_acc: 0.7692

Epoch 00181: val_acc did not improve from 0.84615
Epoch 182/256
 - 6s - loss: 0.9807 - acc: 0.8298 - val_loss: 1.3199 - val_acc: 0.7832

Epoch 00182: val_acc did not improve from 0.84615
Epoch 183/256
 - 6s - loss: 0.9016 - acc: 0.8474 - val_loss: 1.4003 - val_acc: 0.7343

Epoch 00183: val_acc did not improve from 0.84615
Epoch 184/256
 - 7s - loss: 0.9101 - acc: 0.8386 - val_loss: 1.3159 - val_acc: 0.8182

Epoch 00184: val_acc did not improve from 0.84615
Epoch 185/256
 - 7s - loss: 0.9326 - acc: 0.8298 - val_loss: 1.5447 - val_acc: 0.7413

Epoch 00185: val_acc did not improve from 0.84615
Epoch 186/256
 - 5s - loss: 0.9110 - acc: 0.8439 - val_loss: 1.2580 - val_acc: 0.7972

Epoch 00186: val_acc did not improve from 0.84615
Epoch 187/256
 - 6s - loss: 0.8924 - acc: 0.8333 - val_loss: 1.3105 - val_acc: 0.7692

Epoch 00187: val_acc did not improve from 0.84615
Epoch 188/256
 - 7s - loss: 0.9532 - acc: 0.8333 - val_loss: 1.2988 - val_acc: 0.8042

Epoch 00188: val_acc did not improve from 0.84615
Epoch 189/256
 - 6s - loss: 0.8751 - acc: 0.8351 - val_loss: 1.4426 - val_acc: 0.7552

Epoch 00189: val_acc did not improve from 0.84615
Epoch 190/256
 - 5s - loss: 0.9634 - acc: 0.8228 - val_loss: 1.2115 - val_acc: 0.7972

Epoch 00190: val_acc did not improve from 0.84615
Epoch 191/256
 - 6s - loss: 0.9109 - acc: 0.8298 - val_loss: 1.2872 - val_acc: 0.7832

Epoch 00191: val_acc did not improve from 0.84615
Epoch 192/256
 - 5s - loss: 0.9302 - acc: 0.8316 - val_loss: 1.2979 - val_acc: 0.7622

Epoch 00192: val_acc did not improve from 0.84615
Epoch 193/256
 - 6s - loss: 0.9665 - acc: 0.8158 - val_loss: 1.3988 - val_acc: 0.7063

Epoch 00193: val_acc did not improve from 0.84615
Epoch 194/256
 - 6s - loss: 0.8836 - acc: 0.8544 - val_loss: 1.2697 - val_acc: 0.7972

Epoch 00194: val_acc did not improve from 0.84615
Epoch 195/256
 - 6s - loss: 0.8213 - acc: 0.8684 - val_loss: 1.2550 - val_acc: 0.8042

Epoch 00195: val_acc did not improve from 0.84615
Epoch 196/256
 - 6s - loss: 0.8865 - acc: 0.8474 - val_loss: 1.3497 - val_acc: 0.7622

Epoch 00196: val_acc did not improve from 0.84615
Epoch 197/256
 - 5s - loss: 0.9135 - acc: 0.8456 - val_loss: 1.3676 - val_acc: 0.7692

Epoch 00197: val_acc did not improve from 0.84615
Epoch 198/256
 - 5s - loss: 0.9523 - acc: 0.8281 - val_loss: 1.1876 - val_acc: 0.8252

Epoch 00198: val_acc did not improve from 0.84615
Epoch 199/256
 - 6s - loss: 0.8837 - acc: 0.8439 - val_loss: 1.6427 - val_acc: 0.7133

Epoch 00199: val_acc did not improve from 0.84615
Epoch 200/256
 - 7s - loss: 0.9184 - acc: 0.8263 - val_loss: 1.4987 - val_acc: 0.7133

Epoch 00200: val_acc did not improve from 0.84615
Epoch 201/256
 - 6s - loss: 0.9097 - acc: 0.8368 - val_loss: 1.2069 - val_acc: 0.8182

Epoch 00201: val_acc did not improve from 0.84615
Epoch 202/256
 - 6s - loss: 0.9304 - acc: 0.8544 - val_loss: 1.4097 - val_acc: 0.7902

Epoch 00202: val_acc did not improve from 0.84615
Epoch 203/256
 - 6s - loss: 0.9162 - acc: 0.8439 - val_loss: 1.2270 - val_acc: 0.8112

Epoch 00203: val_acc did not improve from 0.84615
Epoch 204/256
 - 6s - loss: 0.9150 - acc: 0.8526 - val_loss: 1.0701 - val_acc: 0.8112

Epoch 00204: val_acc did not improve from 0.84615
Epoch 205/256
 - 6s - loss: 0.8813 - acc: 0.8421 - val_loss: 1.3801 - val_acc: 0.7133

Epoch 00205: val_acc did not improve from 0.84615
Epoch 206/256
 - 6s - loss: 0.8911 - acc: 0.8421 - val_loss: 1.3524 - val_acc: 0.7413

Epoch 00206: val_acc did not improve from 0.84615
Epoch 207/256
 - 6s - loss: 0.8739 - acc: 0.8561 - val_loss: 1.5379 - val_acc: 0.7483

Epoch 00207: val_acc did not improve from 0.84615
Epoch 208/256
 - 6s - loss: 0.8807 - acc: 0.8439 - val_loss: 1.3484 - val_acc: 0.8182

Epoch 00208: val_acc did not improve from 0.84615
Epoch 209/256
 - 6s - loss: 0.9237 - acc: 0.8351 - val_loss: 1.4451 - val_acc: 0.7273

Epoch 00209: val_acc did not improve from 0.84615
Epoch 210/256
 - 6s - loss: 0.9107 - acc: 0.8491 - val_loss: 1.3180 - val_acc: 0.8112

Epoch 00210: val_acc did not improve from 0.84615
Epoch 211/256
 - 5s - loss: 0.9217 - acc: 0.8333 - val_loss: 1.4671 - val_acc: 0.7902

Epoch 00211: val_acc did not improve from 0.84615
Epoch 212/256
 - 6s - loss: 0.9549 - acc: 0.8298 - val_loss: 1.3036 - val_acc: 0.8042

Epoch 00212: val_acc did not improve from 0.84615
Epoch 213/256
 - 6s - loss: 0.9589 - acc: 0.8439 - val_loss: 1.6690 - val_acc: 0.6434

Epoch 00213: val_acc did not improve from 0.84615
Epoch 214/256
 - 5s - loss: 0.9433 - acc: 0.8404 - val_loss: 1.3350 - val_acc: 0.7762

Epoch 00214: val_acc did not improve from 0.84615
Epoch 215/256
 - 6s - loss: 0.8603 - acc: 0.8684 - val_loss: 1.3922 - val_acc: 0.8042

Epoch 00215: val_acc did not improve from 0.84615
Epoch 216/256
 - 6s - loss: 0.9462 - acc: 0.8316 - val_loss: 1.2791 - val_acc: 0.7832

Epoch 00216: val_acc did not improve from 0.84615
Epoch 217/256
 - 6s - loss: 0.8663 - acc: 0.8649 - val_loss: 1.4513 - val_acc: 0.7063

Epoch 00217: val_acc did not improve from 0.84615
Epoch 218/256
 - 6s - loss: 0.9143 - acc: 0.8491 - val_loss: 1.5091 - val_acc: 0.7692

Epoch 00218: val_acc did not improve from 0.84615
Epoch 219/256
 - 6s - loss: 0.8905 - acc: 0.8667 - val_loss: 1.4984 - val_acc: 0.7762

Epoch 00219: val_acc did not improve from 0.84615
Epoch 220/256
 - 5s - loss: 0.9297 - acc: 0.8368 - val_loss: 1.1710 - val_acc: 0.7972

Epoch 00220: val_acc did not improve from 0.84615
Epoch 221/256
 - 6s - loss: 0.8848 - acc: 0.8579 - val_loss: 1.2422 - val_acc: 0.7692

Epoch 00221: val_acc did not improve from 0.84615
Epoch 222/256
 - 6s - loss: 0.9461 - acc: 0.8421 - val_loss: 1.2073 - val_acc: 0.7972

Epoch 00222: val_acc did not improve from 0.84615
Epoch 223/256
 - 6s - loss: 0.8552 - acc: 0.8614 - val_loss: 1.2813 - val_acc: 0.7762

Epoch 00223: val_acc did not improve from 0.84615
Epoch 224/256
 - 6s - loss: 0.9629 - acc: 0.8211 - val_loss: 1.1701 - val_acc: 0.8392

Epoch 00224: val_acc did not improve from 0.84615
Epoch 225/256
 - 6s - loss: 0.9230 - acc: 0.8544 - val_loss: 1.3402 - val_acc: 0.7762

Epoch 00225: val_acc did not improve from 0.84615
Epoch 226/256
 - 6s - loss: 0.9561 - acc: 0.8333 - val_loss: 1.1450 - val_acc: 0.7902

Epoch 00226: val_acc did not improve from 0.84615
Epoch 227/256
 - 6s - loss: 0.8964 - acc: 0.8544 - val_loss: 1.2926 - val_acc: 0.8112

Epoch 00227: val_acc did not improve from 0.84615
Epoch 228/256
 - 5s - loss: 0.8642 - acc: 0.8649 - val_loss: 1.2830 - val_acc: 0.8462

Epoch 00228: val_acc did not improve from 0.84615
Epoch 229/256
 - 5s - loss: 0.8911 - acc: 0.8561 - val_loss: 1.2570 - val_acc: 0.8042

Epoch 00229: val_acc did not improve from 0.84615
Epoch 230/256
 - 6s - loss: 0.9389 - acc: 0.8246 - val_loss: 1.2030 - val_acc: 0.8252

Epoch 00230: val_acc did not improve from 0.84615
Epoch 231/256
 - 6s - loss: 0.9206 - acc: 0.8509 - val_loss: 1.1241 - val_acc: 0.7902

Epoch 00231: val_acc did not improve from 0.84615
Epoch 232/256
 - 6s - loss: 0.8739 - acc: 0.8632 - val_loss: 1.1517 - val_acc: 0.8112

Epoch 00232: val_acc did not improve from 0.84615
Epoch 233/256
 - 6s - loss: 0.8873 - acc: 0.8579 - val_loss: 1.1664 - val_acc: 0.8112

Epoch 00233: val_acc did not improve from 0.84615
Epoch 234/256
 - 5s - loss: 0.9306 - acc: 0.8316 - val_loss: 1.2338 - val_acc: 0.8042

Epoch 00234: val_acc did not improve from 0.84615
Epoch 235/256
 - 6s - loss: 0.8566 - acc: 0.8596 - val_loss: 1.3431 - val_acc: 0.7692

Epoch 00235: val_acc did not improve from 0.84615
Epoch 236/256
 - 6s - loss: 0.9113 - acc: 0.8404 - val_loss: 1.3116 - val_acc: 0.7972

Epoch 00236: val_acc did not improve from 0.84615
Epoch 237/256
 - 6s - loss: 0.8646 - acc: 0.8649 - val_loss: 1.2718 - val_acc: 0.8182

Epoch 00237: val_acc did not improve from 0.84615
Epoch 238/256
 - 5s - loss: 0.8817 - acc: 0.8579 - val_loss: 1.1519 - val_acc: 0.7902

Epoch 00238: val_acc did not improve from 0.84615
Epoch 239/256
 - 7s - loss: 0.9028 - acc: 0.8456 - val_loss: 1.2130 - val_acc: 0.8042

Epoch 00239: val_acc did not improve from 0.84615
Epoch 240/256
 - 7s - loss: 0.9452 - acc: 0.8263 - val_loss: 1.2195 - val_acc: 0.8042

Epoch 00240: val_acc did not improve from 0.84615
Epoch 241/256
 - 6s - loss: 0.9060 - acc: 0.8509 - val_loss: 1.4104 - val_acc: 0.7552

Epoch 00241: val_acc did not improve from 0.84615
Epoch 242/256
 - 6s - loss: 0.8737 - acc: 0.8596 - val_loss: 1.4080 - val_acc: 0.7692

Epoch 00242: val_acc did not improve from 0.84615
Epoch 243/256
 - 5s - loss: 0.8931 - acc: 0.8667 - val_loss: 1.2159 - val_acc: 0.8182

Epoch 00243: val_acc did not improve from 0.84615
Epoch 244/256
 - 6s - loss: 0.8216 - acc: 0.8842 - val_loss: 1.3687 - val_acc: 0.7413

Epoch 00244: val_acc did not improve from 0.84615
Epoch 245/256
 - 5s - loss: 0.9006 - acc: 0.8737 - val_loss: 1.2637 - val_acc: 0.8042

Epoch 00245: val_acc did not improve from 0.84615
Epoch 246/256
 - 6s - loss: 0.8297 - acc: 0.8772 - val_loss: 1.1375 - val_acc: 0.8112

Epoch 00246: val_acc did not improve from 0.84615
Epoch 247/256
 - 6s - loss: 0.8750 - acc: 0.8561 - val_loss: 1.2408 - val_acc: 0.8182

Epoch 00247: val_acc did not improve from 0.84615
Epoch 248/256
 - 6s - loss: 0.8754 - acc: 0.8667 - val_loss: 1.5848 - val_acc: 0.7552

Epoch 00248: val_acc did not improve from 0.84615
Epoch 249/256
 - 6s - loss: 0.9242 - acc: 0.8561 - val_loss: 1.3895 - val_acc: 0.7483

Epoch 00249: val_acc did not improve from 0.84615
Epoch 250/256
 - 6s - loss: 0.8902 - acc: 0.8544 - val_loss: 1.2786 - val_acc: 0.7902

Epoch 00250: val_acc did not improve from 0.84615
Epoch 251/256
 - 7s - loss: 0.8544 - acc: 0.8667 - val_loss: 1.2658 - val_acc: 0.7692

Epoch 00251: val_acc did not improve from 0.84615
Epoch 252/256
 - 6s - loss: 0.9343 - acc: 0.8561 - val_loss: 1.3585 - val_acc: 0.7972

Epoch 00252: val_acc did not improve from 0.84615
Epoch 253/256
 - 5s - loss: 0.8510 - acc: 0.8860 - val_loss: 1.6780 - val_acc: 0.6993

Epoch 00253: val_acc did not improve from 0.84615
Epoch 254/256
 - 6s - loss: 0.8533 - acc: 0.8702 - val_loss: 1.2169 - val_acc: 0.7762

Epoch 00254: val_acc did not improve from 0.84615
Epoch 255/256
 - 6s - loss: 0.8645 - acc: 0.8526 - val_loss: 1.2667 - val_acc: 0.7972

Epoch 00255: val_acc did not improve from 0.84615
Epoch 256/256
 - 6s - loss: 0.8424 - acc: 0.8754 - val_loss: 1.4055 - val_acc: 0.7552

Epoch 00256: val_acc did not improve from 0.84615
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
Test Loss: 1.0267
Test Accuracy: 0.8175

Confusion Matrix: 
[[ 1  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0 12  0  4  0  0  0  0  0  0  0  0  1]
 [ 0  0  1  0  1  0  0  0  0  0  0  0  0]
 [ 0  3  0  8  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0 15  0  0  0  1  0  0  1  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  9  0  0  0  1  0  0]
 [ 0  0  0  0  1  0  0  4  0  2  0  0  0]
 [ 0  0  0  0  3  0  0  0 12  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0 13  0  0  0]
 [ 0  1  0  0  0  0  3  0  0  0 16  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  1  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0 11]]

Classification Report: 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         1
           1       0.75      0.71      0.73        17
           2       1.00      0.50      0.67         2
           3       0.67      0.73      0.70        11
           4       0.75      0.88      0.81        17
           5       0.00      0.00      0.00         1
           6       0.00      0.00      0.00         0
           7       0.75      0.90      0.82        10
           8       1.00      0.57      0.73         7
           9       0.92      0.80      0.86        15
          10       0.87      1.00      0.93        13
          11       0.94      0.80      0.86        20
          12       0.50      1.00      0.67         1
          13       0.92      1.00      0.96        11

    accuracy                           0.82       126
   macro avg       0.72      0.71      0.69       126
weighted avg       0.83      0.82      0.82       126


Training Time: 1568.3427 s s
```
### Metadata

{
    "input_filepath": "./data/",
    "output_filepath": "./output/",
    "train_test_split": 0.85,
    "sequence_length": 1024,
    "labels": [
        "Brush_teeth",
        "Climb_stairs",
        "Comb_hair",
        "Descend_stairs",
        "Drink_glass",
        "Eat_meat",
        "Eat_soup",
        "Getup_bed",
        "Liedown_bed",
        "Pour_water",
        "Sitdown_chair",
        "Standup_chair",
        "Use_telephone",
        "Walk"
    ],
    "num_categories": 14,
    "class_samples": {
        "Brush_teeth": 12,
        "Climb_stairs": 102,
        "Comb_hair": 31,
        "Descend_stairs": 42,
        "Drink_glass": 100,
        "Eat_meat": 5,
        "Eat_soup": 3,
        "Getup_bed": 101,
        "Liedown_bed": 28,
        "Pour_water": 100,
        "Sitdown_chair": 100,
        "Standup_chair": 102,
        "Use_telephone": 13,
        "Walk": 100
    },
    "total_samples": 839,
    "class_frequencies": {
        "Brush_teeth": 0.014303,
        "Climb_stairs": 0.121573,
        "Comb_hair": 0.036949,
        "Descend_stairs": 0.05006,
        "Drink_glass": 0.11919,
        "Eat_meat": 0.005959,
        "Eat_soup": 0.003576,
        "Getup_bed": 0.120381,
        "Liedown_bed": 0.033373,
        "Pour_water": 0.11919,
        "Sitdown_chair": 0.11919,
        "Standup_chair": 0.121573,
        "Use_telephone": 0.015495,
        "Walk": 0.11919
    },
    "shapes": {
        "x": [
            839,
            1024,
            3
        ],
        "y": [
            839,
            14
        ]
    },
    "epochs": 256,
    "batch_size": 4,
    "train_val_split": 0.2,
    "train_time": "1568.3427 s",
    "mean_epoch_train_time": "6.1263 s"
}
