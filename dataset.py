import torch
import torch.utils.data as data_utils
from torch.utils.data import random_split, DataLoader
import SimpleITK as sitk


class ADNIDataset(data_utils.Dataset):
    def __init__(self, data_list, label_list, is_training=True):
        self.data_list, self.label_list = data_list, label_list
        self.is_training = is_training

    def __len__(self):
        return len(self.data_list)

    def load(self, image_path):
        # print(image_path)
        vol = sitk.Image(sitk.ReadImage(image_path))
        # inputsize = vol.GetSize()
        # print(f'inputsize{inputsize}')
        # inputspacing = vol.GetSpacing()
        # print(f'inputspacing{inputspacing}')
        if self.is_training == True:
            rand1 = torch.rand(1)[0]
            if rand1 >= 0 and rand1 < 0.166:
                # print('Option 1')
                img_array = sitk.GetArrayFromImage(vol)[0:160, 12:204, 10:170]
            elif rand1 >= 0.166 and rand1 < 0.333:
                img_array = sitk.GetArrayFromImage(vol)[2:162, 12:204, 10:170]
                # print('Option 2')
            elif rand1 >= 0.333 and rand1 < 0.499:
                img_array = sitk.GetArrayFromImage(vol)[1:161, 11:203, 10:170]
                # print('Option 3')
            elif rand1 >= 0.499 and rand1 < 0.666:
                img_array = sitk.GetArrayFromImage(vol)[1:161, 13:205, 10:170]
                # print('Option 4')
            elif rand1 >= 0.666 and rand1 < 0.833:
                img_array = sitk.GetArrayFromImage(vol)[1:161, 12:204, 9:169]
                # print('Option 5')
            else:
                img_array = sitk.GetArrayFromImage(vol)[1:161, 12:204, 11:171]
                # print('Option 6')
        else:
            # Test phase
            img_array = sitk.GetArrayFromImage(vol)[1:161, 12:204, 10:170]

        # Mirror left and right brain
        if self.is_training == True:
            rand = torch.rand(1)[0]
            if rand > 0.5:
                img_array = img_array[:, :, ::-1]
        img_array = img_array / img_array.max()
        return torch.Tensor(img_array).unsqueeze(dim=0)

    def __getitem__(self, index):
        bag_path = self.data_list[index]
        label = self.label_list[index]
        tensor = self.load(bag_path)
        return tensor, int(label)


if __name__ == "__main__":
    bags_list = []
    labels_list = []
    with open(r'AD_CN_index.txt', encoding="utf-8") as file:
        content = file.readlines()
        # Read data line by line
        for line in content:
            bags_list.append(line.split('   ')[0])
            labels_list.append(line.split('   ')[1].replace('\n', ''))
    dataset = ADNIDataset(bags_list, labels_list)
    print('The total length of the dataset', len(dataset))
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    print('The length of the validation set', len(val_set))
    print('The length of the training set', len(train_set))
    train_loader = DataLoader(train_set, shuffle=True)

    for (data, label) in train_loader:
        # print(data.max())
        print(data.shape)
