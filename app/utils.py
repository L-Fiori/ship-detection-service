from skimage.morphology import binary_opening, disk, label
from skimage.measure import label, regionprops
from skimage.io import imread, imshow, imsave
from shapely.geometry import Polygon
from keras.models import load_model
from matplotlib.cm import get_cmap
from cbers4asat import Cbers4aAPI
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import cm
from datetime import date
from osgeo import gdal
from io import BytesIO
from PIL import Image
import pandas as pd
import numpy as np
import rasterio
import base64
import random
import cv2
import os
from app.models import db, images, ships

def get_api():
    api = Cbers4aAPI("gustavotorrico@usp.br")

    return api

def get_products(loc, start_date, end_date, cloud):
    #provider = 'inpe'
    #if (PROVIDER == 'inpe'):
        #api = Cbers4aAPI(email)
    # satellite = 'cbers'
    api = get_api()
    product_type = 'CBERS4A_WPM_L4_DN'

    extent = get_extent(loc)

    products = api.query(location=extent,
                     initial_date=date(int(start_date.split('-')[0]), int(start_date.split('-')[1]), int(start_date.split('-')[2])),
                     end_date=date(int(end_date.split('-')[0]), int(end_date.split('-')[1]), int(end_date.split('-')[2])),
                     cloud=cloud,
                     limit=3,
                     collections=[product_type])
    print('products: ', len(products['features']))

    return products

def get_extent(loc):
    match loc:
      case 'salvador':
        extent = Polygon([
                [-38.51505860581503, -12.978941602226499],
                [-38.51505860581503, -13.030635291915374],
                [-38.454543348785876, -13.030635291915374],
                [-38.454543348785876, -12.978941602226499],
            ])
      case 'recife':
        extent = Polygon([
                [-34.87737558484915, -8.071411029529315],
                [-34.87737558484915, -8.056976194056404],
                [-34.854444033281595, -8.056976194056404],
                [-34.854444033281595, -8.071411029529315],
            ])
      case 'santos':
        extent = Polygon([
                [-46.34258562906814, -23.973693250698503],
                [-46.34258562906814, -23.984440384202856],
                [-46.32457462294434, -23.984440384202856],
                [-46.32457462294434, -23.973693250698503],
            ])
      case 'vitoria':
        extent = Polygon([
                [-40.27913147141851, -20.315899721748366],
                [-40.27913147141851, -20.31793686489486],
                [-40.27643483685614, -20.31793686489486],
                [-40.27643483685614, -20.315899721748366],
            ])

    return extent

def get_model():
    model_path = os.path.join('./app/models', 'fullres_model.h5')
    #model_path = os.path.abspath(model_path)
    model = load_model(model_path)
    return model

def download_images(products):
    api = get_api()
    fp_in_images = os.path.join(".", "images")
    products_sample = products.copy()
    product_counter = 0

    for sample in products['features']:
        products_sample['features'] = [products['features'][product_counter]]
        product_counter += 1
        sub = None

        # Get download image folder name
        fp_sample = os.path.join(fp_in_images, sample['id'])
        print(os.path.abspath(fp_sample), os.path.abspath(fp_in_images))

        # Download image folder if not already there
        if not os.path.isdir(fp_sample):
          api.download(products=products_sample,
                    bands=['red', 'green', 'blue', 'nir'],
                    threads=6,  # Numero de downloads simultâneos
                    outdir=fp_in_images,
                    with_folder=True)

def run(products, location):
    fp_in_images = os.path.join(".", "images")
    
    for sample in products['features']:
        fp_sample = os.path.join(fp_in_images, sample['id'])
        # Get folder paths of band folders
        for filename in os.listdir(fp_sample):
            if filename.endswith("BAND1.tif"):
                fp_band_blue = fp_sample + '/' + filename
            if filename.endswith("BAND2.tif"):
                fp_band_green = fp_sample + '/' + filename
            if filename.endswith("BAND3.tif"):
                fp_band_red = fp_sample + '/' + filename
            if filename.endswith("BAND4.tif"):
                fp_band_nir = fp_sample + '/' + filename

        print(sample['id'])
        
        # Run functions
        num_tiles_horizontally, num_tiles_vertically = cut_tif_into_tiles(fp_band_blue, fp_band_green, fp_band_red, fp_band_nir, sample['id'])
        #num_tiles_horizontally, num_tiles_vertically = 19, 19

        get_rgb_images(num_tiles_horizontally, num_tiles_vertically, sample['id'])
        split_and_resize_images('split_once', sample['id']) # no_split, split_once
        model = get_model()
        sub = predict_ships(sample['id'], model)
        add_to_database(sub, sample['id'], 'split_once', model, location)

        break

#=======================================================

def predict(img, model, image_dir):
    c_img = imread(os.path.join(image_dir, img))
    c_img = np.expand_dims(c_img, 0) / 255.0
    cur_seg = model.predict(c_img)[0]
    cur_seg = binary_opening(cur_seg > 0.9, np.expand_dims(disk(2), -1))
    return cur_seg, c_img

def multi_rle_encode(img, **kwargs):
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]
    else:
        return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def pred_encode(img, model, img_path, **kwargs):
    cur_seg, _ = predict(img, model, img_path)
    cur_rles = multi_rle_encode(cur_seg, **kwargs)
    return [[img, rle] for rle in cur_rles if rle is not None]

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_color(in_mask_list):
    # Take the individual ship masks and create a color mask array for each ships
    all_masks = np.zeros((len(in_mask_list), 768, 768))
    scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2) ## scale the heatmap image to shift
    for i,mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[i] = np.zeros((768, 768), dtype = float) + scale(i) * rle_decode(mask)
    return all_masks

#=======================================================

def cut_tif_into_tiles(fp_band_blue, fp_band_green, fp_band_red, fp_band_nir, sample_id):
  fp_tiles = os.path.join(os.path.join('./images', sample_id), 'tiles')
  fp_blue = os.path.join(fp_tiles, 'blue/')
  fp_green = os.path.join(fp_tiles, 'green/')
  fp_red = os.path.join(fp_tiles, 'red/')
  fp_nir = os.path.join(fp_tiles, 'nir/')

  if not os.path.isdir(fp_tiles): os.mkdir(fp_tiles)
  if not os.path.isdir(fp_blue): os.mkdir(fp_blue)
  if not os.path.isdir(fp_green): os.mkdir(fp_green)
  if not os.path.isdir(fp_red): os.mkdir(fp_red)
  if not os.path.isdir(fp_nir): os.mkdir(fp_nir)

  fn_blue = 'tile_blue_'
  fn_green = 'tile_green_'
  fn_red = 'tile_red_'
  fn_nir = 'tile_nir_'

  tile_size_x = 768
  tile_size_y = 768
  count_x = 0
  count_y = 0
  count = 0

  # Band 1 - Blue
  ds = gdal.Open(fp_band_blue)
  band = ds.GetRasterBand(1)
  xsize = band.XSize
  ysize = band.YSize
  size = min(xsize, ysize)
  for j in range(0, size, tile_size_y):
      count_x = 0
      for i in range(0, size, tile_size_x):
          gdal.Translate(str(fp_blue) + str(fn_blue) + str(count) + ".tif", ds, srcWin = (i, j, tile_size_x, tile_size_y))
          count += 1
          count_x += 1 # num linhas
      count_y += 1 # num colunas
  count = 0
  # Band 2 - Green
  ds = gdal.Open(fp_band_green)
  band = ds.GetRasterBand(1)
  xsize = band.XSize
  ysize = band.YSize
  size = min(xsize, ysize)
  for j in range(0, size, tile_size_y):
      for i in range(0, size, tile_size_x):
          gdal.Translate(str(fp_green) + str(fn_green) + str(count) + ".tif", ds, srcWin = (i, j, tile_size_x, tile_size_y))
          count += 1
  count = 0
  # Band 3 - Red
  ds = gdal.Open(fp_band_red)
  band = ds.GetRasterBand(1)
  xsize = band.XSize
  ysize = band.YSize
  size = min(xsize, ysize)
  for j in range(0, size, tile_size_y):
      for i in range(0, size, tile_size_x):
          gdal.Translate(str(fp_red) + str(fn_red) + str(count) + ".tif", ds, srcWin = (i, j, tile_size_x, tile_size_y))
          count += 1
  count = 0
  # Band 4 - Nir
  ds = gdal.Open(fp_band_nir)
  band = ds.GetRasterBand(1)
  xsize = band.XSize
  ysize = band.YSize
  size = min(xsize, ysize)
  for j in range(0, size, tile_size_y):
      for i in range(0, size, tile_size_x):
          gdal.Translate(str(fp_nir) + str(fn_nir) + str(count) + ".tif", ds, srcWin = (i, j, tile_size_x, tile_size_y))
          count += 1
  count = 0

  print(count_x, count_y)
  return count_x, count_y

#=======================================================

def split_and_resize_images(split, sample_id):
  fp_tiles = os.path.join(os.path.join('./images', sample_id), 'tiles')
  fp_rgb = os.path.join(fp_tiles, 'rgb/')
  for img_name in os.listdir(fp_rgb):
    # Read
    img_path = fp_rgb + img_name
    if '.tif' in img_path:
      band_ndwi = rasterio.open(img_path)
      image = band_ndwi.read(1)
      width, height = image.shape
    if '.jpg' in img_path:
      image = imread(img_path)
      width, height, layers = image.shape

    # Crop
    match split:
      case 'no_split':
        # Resize
        resized = np.array(Image.fromarray((image * 255).astype(np.uint8)).resize((768, 768)))
        if '.jpg' in img_path: resized = cv2.bitwise_not(resized)

        # Save
        os.remove(img_path)
        imsave(img_path, resized)

      case 'split_once':
        crop1 = image[0:height//2,      0:width//2]
        crop2 = image[0:height//2,      width//2:width]
        crop3 = image[height//2:height, 0:width//2]
        crop4 = image[height//2:height, width//2:width]

        # Resize
        crop1_resized = np.array(Image.fromarray((crop1 * 255).astype(np.uint8)).resize((768, 768)))
        crop2_resized = np.array(Image.fromarray((crop2 * 255).astype(np.uint8)).resize((768, 768)))
        crop3_resized = np.array(Image.fromarray((crop3 * 255).astype(np.uint8)).resize((768, 768)))
        crop4_resized = np.array(Image.fromarray((crop4 * 255).astype(np.uint8)).resize((768, 768)))

        if '.jpg' in img_path:
          crop1_resized = cv2.bitwise_not(crop1_resized)
          crop2_resized = cv2.bitwise_not(crop2_resized)
          crop3_resized = cv2.bitwise_not(crop3_resized)
          crop4_resized = cv2.bitwise_not(crop4_resized)

        # Save
        if '.tif' in img_path:
          img_name_formatted = img_path.replace(".tif", "")
          extension = '.tif'
        if '.jpg' in img_path:
          img_name_formatted = img_path.replace(".jpg", "")
          extension = '.jpg'
        imsave(img_name_formatted + '_1' + extension, crop1_resized)
        imsave(img_name_formatted + '_2' + extension, crop2_resized)
        imsave(img_name_formatted + '_3' + extension, crop3_resized)
        imsave(img_name_formatted + '_4' + extension, crop4_resized)
        os.remove(img_path)

#=======================================================

def predict_ships(sample_id, model):
  out_pred_rows = []
  fp_tiles = os.path.join(os.path.join('./images', sample_id), 'tiles')
  fp_rgb = os.path.join(fp_tiles, 'rgb/')
  for img_name in os.listdir(fp_rgb):
    if 'ndwi' not in img_name:
      print("Predicting ships in image ", img_name)
      out_pred_rows += pred_encode(img_name, model, fp_rgb, min_max_threshold=1.0)

  if (out_pred_rows != []):
    sub = pd.DataFrame(out_pred_rows)
    sub.columns = ['ImageId', 'EncodedPixels']
    sub = sub[sub.EncodedPixels.notnull()]
    if (len(sub.index) == 1): sub = pd.concat([sub,sub.loc[[0],:]]).sort_index() # Make copy if single row
    return sub
  else:
    return None

#=======================================================

def analyse_predictions(img, array_of_ships, prop, ndwi):
  x1, y1, x2, y2 = prop.bbox[1], prop.bbox[0],  prop.bbox[3], prop.bbox[2]
  if x1 >= 768: x1 = 767
  if y1 >= 768: y1 = 767
  if x2 >= 768: x2 = 767
  if y2 >= 768: y2 = 767
  if ndwi[y1][x1] == 0: return False
  if ndwi[y1][x2] == 0: return False
  if ndwi[y2][x1] == 0: return False
  if ndwi[y2][x2] == 0: return False
  return True

def raw_prediction(img, model, image_dir):
    c_img = imread(os.path.join(image_dir, img))
    c_img = np.expand_dims(c_img, 0)/255.0
    cur_seg = model.predict(c_img)[0]
    return cur_seg, c_img[0]

def add_to_database(sub, file_name, split, model, location):
  if type(sub) == pd.core.frame.DataFrame:
    for img in sub.ImageId.unique():
      print("Processing image: ", img)

      fp_tiles = os.path.join(os.path.join('./images', file_name), 'tiles')
      img_path = os.path.join(fp_tiles, 'rgb/')
      pred, c_img = raw_prediction(img, model, img_path)

      # Ship masks
      array_of_ships = masks_as_color(sub.query('ImageId==\"{}\"'.format(img))['EncodedPixels'])

      # Get ndwi
      img_ndwi = img.replace('RGB', 'ndwi')
      img_ndwi = img_ndwi.replace('.jpg', '.tif')
      band_ndwi = rasterio.open(img_path + img_ndwi)
      ndwi = band_ndwi.read(1)
      ndwi_base = np.copy(ndwi)
      ndwi[ndwi_base <= 0.0] = 0 # = land
      ndwi[ndwi_base > 0.0] = 1 # = water

      # Get bounding boxes
      lbl_0 = label(pred[...,0])
      props = regionprops(lbl_0)
      img_1 = c_img.copy()
      num_of_ships = 0
      ship_num_arr = []
      ship_size_arr = []
      ship_class_arr = []
      for prop in props:
          if analyse_predictions(img, array_of_ships, prop, ndwi):
            cv2.rectangle(img_1, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (0, 255, 0), 2)
            num_of_ships += 1
            ship_num_arr.append(num_of_ships)
            ship_size_arr.append(random.randint(150, 400))
            ship_class_arr.append("cargo")
          else:
            cv2.rectangle(img_1, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)

      # Save and get base64
      img_wo_extension = img.replace(".jpg", "")
      fp_output = './images/output/'
      if not os.path.isdir(fp_output): os.mkdir(fp_output)
      file_path = fp_output + img_wo_extension + '_' + split + '.jpg'
      plt.imshow(img_1)
      plt.axis('off')
      plt.savefig(file_path, dpi=200, bbox_inches='tight', pad_inches=0)
      plt.close()

      pil_img = Image.open(file_path)
      buff = BytesIO()
      pil_img.save(buff, format="JPEG")
      pil_img_64 = base64.b64encode(buff.getvalue())

      # Add to database
      datetime_str = '09/19/22 13:55:26'
      datetime_object = datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')

      if images.query.filter_by(image = pil_img_64).first() is None:
        imgs = images(img, pil_img_64, num_of_ships, file_name, datetime_object, location)
        db.session.add(imgs)
        db.session.commit()
        img_id = images.query.filter_by(image = pil_img_64).first()._id
        for ship_count in range(0,num_of_ships):
          ship = ships(ship_num_arr[ship_count], ship_class_arr[ship_count], ship_size_arr[ship_count], img_id)
          db.session.add(ship)
          db.session.commit()

#=======================================================

def plot_sized_predictions(sub, file_name, split, model):
  print('Total ships in all images: ', len(sub.index))
  TOP_PREDICTIONS = len(sub.index)
  num_of_samples = 4
  if (TOP_PREDICTIONS > num_of_samples): TOP_PREDICTIONS = num_of_samples # = num_of_samples
  fig, m_axs = plt.subplots(TOP_PREDICTIONS, 3, figsize = (12, TOP_PREDICTIONS*4))
  [c_ax.axis('off') for c_ax in m_axs.flatten()]

  def analyse_predictions(img, array_of_ships, prop, ndwi):
    x1, y1, x2, y2 = prop.bbox[1], prop.bbox[0],  prop.bbox[3], prop.bbox[2]
    if x1 >= 768: x1 = 767
    if y1 >= 768: y1 = 767
    if x2 >= 768: x2 = 767
    if y2 >= 768: y2 = 767
    if ndwi[y1][x1] == 0: return False
    if ndwi[y1][x2] == 0: return False
    if ndwi[y2][x1] == 0: return False
    if ndwi[y2][x2] == 0: return False
    return True

  def raw_prediction(img, model, image_dir):
      c_img = imread(os.path.join(image_dir, img))
      c_img = np.expand_dims(c_img, 0)/255.0
      cur_seg = model.predict(c_img)[0]
      return cur_seg, c_img[0]

  first = True
  for (ax1, ax2, ax3), img in zip(m_axs, sub.ImageId.unique()[:TOP_PREDICTIONS]):
      # Get prediction
      fp_tiles = os.path.join(os.path.join('./images', file_name), 'tiles')
      img_path = os.path.join(fp_tiles, 'rgb/')
      pred, c_img = raw_prediction(img, model, img_path)
      print("Img: ", img)
      pil_img = Image.open(os.path.join(img_path, img))

      array_of_ships = masks_as_color(sub.query('ImageId==\"{}\"'.format(img))['EncodedPixels'])

      # Get ndwi
      img_ndwi = img.replace('RGB', 'ndwi')
      img_ndwi = img_ndwi.replace('.jpg', '.tif')
      band_ndwi = rasterio.open(img_path + img_ndwi)
      ndwi = band_ndwi.read(1)
      ndwi_base = np.copy(ndwi)
      ndwi[ndwi_base <= 0.0] = 0 # = land
      ndwi[ndwi_base > 0.0] = 1 # = water

      # Get bounding boxes
      lbl_0 = label(pred[...,0])
      props = regionprops(lbl_0)
      img_1 = c_img.copy()
      for prop in props:
          print('Found bbox ', prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2])

          if analyse_predictions(img, array_of_ships, prop, ndwi):
            cv2.rectangle(img_1, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (0, 255, 0), 2)
          else:
            cv2.rectangle(img_1, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)

      ax1.imshow(c_img)
      if first: ax1.set_title('Image', fontsize=18)#('Image: ' + img, fontsize=15)
      ax2.imshow(np.sum(np.transpose(array_of_ships, (1, 2, 0)), 2), cmap=get_cmap('jet'))
      if first: ax2.set_title('Ship mask', fontsize=18)
      ax3.imshow(img_1)
      if first: ax3.set_title('Bounding boxes', fontsize=18)
      first = False

      fp_output = './images/output/'
      if not os.path.isdir(fp_output): os.mkdir(fp_output)
      fig.savefig(fp_output + file_name + '_' + split + '.jpg', dpi=200, bbox_inches='tight', pad_inches=0)
      plt.close('all')

#=======================================================

def get_rgb_images(count_x, count_y, sample_id):
  #!rm -rf /content/tiles/rgb
  fp_tiles = os.path.join(os.path.join('./images', sample_id), 'tiles')
  fp_rgb = os.path.join(fp_tiles, 'rgb/')
  if not os.path.isdir(fp_rgb): os.mkdir(fp_rgb)

  fp_blue = os.path.join(fp_tiles, 'blue/')
  fp_green = os.path.join(fp_tiles, 'green/')
  fp_red = os.path.join(fp_tiles, 'red/')
  fp_nir = os.path.join(fp_tiles, 'nir/')

  fig = plt.figure()
  count1 = 0
  count2 = 0
  list_of_images = []

  for filename in os.listdir(fp_blue):
    band_2 = rasterio.open(fp_blue + filename)
    filename = filename.replace("blue", "green")
    band_3 = rasterio.open(fp_green + filename)
    filename = filename.replace("green", "red")
    band_4 = rasterio.open(fp_red + filename)
    filename = filename.replace("red", "nir")
    band_5 = rasterio.open(fp_nir + filename)

    num = filename.replace("tile_nir_", "")
    num = num.replace(".tif", "")
    num = int(num)

    nir = band_5.read(1)
    red = band_4.read(1)
    green = band_3.read(1)
    blue = band_2.read(1)
    rgb_composite_gn = np.dstack((red, green, blue))

    ndwi = (green - nir) / (green + nir)
    ndwi_mean = np.mean(ndwi)
    print("File name: ", filename)
    print("Calculating mean NDWI: ", ndwi_mean)
    if (ndwi_mean > 0.0):
      # Save RGB jpg
      rgb_plot=plt.imshow(rgb_composite_gn, interpolation='lanczos')
      plt.axis('off')
      filename = filename.replace("nir", "RGB")
      filename = filename.replace(".tif", ".jpg")
      plt.savefig(fp_rgb + filename, dpi=200, bbox_inches='tight', pad_inches=0)
      plt.close('all')

      # Save NDWI tif
      filename = filename.replace("RGB", "ndwi")
      filename = filename.replace(".jpg", ".tif")
      ndwi_img = np.copy(ndwi)
      ndwi_img[ndwi <= 0.0] = 0 # = land
      ndwi_img[ndwi > 0.0] = 1 # = water
      cv2.imwrite(fp_rgb + filename, ndwi_img)
