import librosa
import csv
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import torch
import librosa

from torch.utils.data import Dataset


class AudioDataset(Dataset):

    def __init__(self, index, DataRoot, LabelPath, feature="spec", mode="train"):
        """
        feature: 
            "mel": mel spectrum
            "mfcc": mfcc
        """
        self.DataRoot = DataRoot
        self.DataPath = os.path.join(DataRoot, "audio")
        self.FeaturePath = os.path.join(DataRoot, feature)
        self.LabelPath = LabelPath

        self.LabelDict = self.load_label(self.LabelPath)
        
        self.FoldNum = 10
        self.Folds = ["fold{}".format(i) for i in range(1,11)]
        
        if not self.verify():
            print("verify {} feature fail".format(feature))
            self.save_feature(feature)
        print("verify {} feature success".format(feature))

        if mode == "train":
            self.SelectFolds = self.Folds.copy()
            self.SelectFolds.remove("fold{}".format(index))
        else:
            self.SelectFolds = ["fold{}".format(index)]

        self.Audios, self.Labels = self.load(self.SelectFolds)
        
    def load_label(self, LabelPath):

        # load label
        LabelDict = {}
        with open(LabelPath) as csvfile:
            reader = csv.reader(csvfile)
            raw = list(reader)

            for row in raw[1:]:
                LabelDict[row[0][:-4]] = int(row[-2])

        return LabelDict

    def verify(self):

        Len = 0
        if os.path.exists(self.FeaturePath):
            
            for Fold in self.Folds:
                
                FoldPath = os.path.join(self.FeaturePath, Fold)
                
                if os.path.exists(FoldPath):
                    Len += len(os.listdir(FoldPath))
            #print(Len)
            if Len > 8700:
                return True
        return False

    def save_feature(self, feature="mel"):
        Folds = ["fold{}".format(i) for i in range(1,11)]
        if not os.path.exists(self.FeaturePath):
            os.mkdir(self.FeaturePath)
        for Fold in Folds:
            TargetFoldPath = os.path.join(self.FeaturePath, Fold)
            if not os.path.exists(TargetFoldPath):
                os.mkdir(TargetFoldPath)

            FoldPath = os.path.join(self.DataPath, Fold)
            for AudioName in os.listdir(FoldPath):

                # Converted filename will be same as original file, with a different extension
                filename  = os.path.join(TargetFoldPath, AudioName)[:-4]
                filename += ".png"

                if AudioName == ".DS_Store" or os.path.exists(filename):
                    print("skip: {}".format(filename))
                    continue

                print(filename)

                AudioPath = os.path.join(FoldPath, AudioName)

                # Load the audio file as a waveform, store its sampling rate
                samples, sample_rate = librosa.load(AudioPath)

                # Define the dimensions of the spectrogram to be saved
                fig = plt.figure(figsize=[0.72,0.72])
                ax = fig.add_subplot(111)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                ax.set_frame_on(False)

                
                if feature == "spec":
                    S = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
                    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

                elif feature == "mfcc":
                    mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate)
                    librosa.display.specshow(mfcc, x_axis='time')
                else:
                    raise ValueError('Unknown feature type.')
                    

                # Convert the scaling of the spectrogram to dB units

                # Save the converted image 
                plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)

                # Close the open image
                plt.close('all')


    def load(self, Folds):
        # load sound
        Audios = []
        Labels = []

        for Fold in Folds:
            
            FoldPath = os.path.join(self.FeaturePath, Fold)
            for ImgName in os.listdir(FoldPath):
                
                if not ImgName[-4:] == ".png":
                    continue
                       
                ImgPath = os.path.join(FoldPath, ImgName)
                
                Audios.append(ImgPath)
                Labels.append(self.LabelDict[ImgName[:-4]])
            
        return Audios, Labels
    
    def __getitem__(self, idx):
        
        Img = cv2.imread(self.Audios[idx], cv2.IMREAD_COLOR)

        
        Label = self.Labels[idx]
            
        return Img, Label
    
    def __len__(self):
        
        return len(self.Audios)