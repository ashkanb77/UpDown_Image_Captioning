! mkdir 'dataset'

! wget 'http://images.cocodataset.org/zips/train2014.zip'
! unzip -q train2014.zip -d 'dataset'
! rm 'train2014.zip'

! wget 'http://images.cocodataset.org/zips/val2014.zip'
! unzip -q val2014.zip -d 'dataset'
! rm 'val2014.zip'

! wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
! unzip 'annotations_trainval2014.zip' -d 'dataset'
! rm 'annotations_trainval2014.zip'
