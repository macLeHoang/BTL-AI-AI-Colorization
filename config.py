BATCH = 16 # batch size
pretrained_epochs = 20
epochs = 20

path_ = '' # path to your training data

pretrained_weight_path_ = '' # path to folder that saving pre-trained weights
gen_weight_path_ = '' # path to folder that saving generator weights
dis_weight_path_ = '' # path to folder that saving discriminator weights

logs = False # logging losses value during training or not
logs_path = '' # Ignore this if logs is False

############ FOR TEST ##############
# ignore the above configs if use test only

weight_ = ''   # path to weight files
               # E.g: file data: generator-20220630-104911.data-00000-of-00001
               #      file index: generator-20220630-104911.index
               # path to load is: .../generator-20220630-104911
