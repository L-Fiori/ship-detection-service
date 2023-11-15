# Satellite parameters
PROVIDER = 'inpe'
SATELLITE = 'cbers'
LOCATION = 'santos' # salvador, recife, santos, vitoria
START_DATE = '2016-01-01'
END_DATE = '2023-11-11'

#=======================================================

# Credentials
email = "gustavotorrico@usp.br"

if (PROVIDER == 'inpe'):
  api = Cbers4aAPI(email)

#=======================================================

match SATELLITE:
  case 'cbers':
    product_type = 'CBERS4A_WPM_L4_DN'

match LOCATION:
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

#=======================================================

# Potencialmente so se aplica pra ipynb

"""
setup_logging(verbose=3)

# Create the workspace folder
fp_in_images = '/content/input/images/'
if not os.path.isdir(fp_in_images): os.mkdir(fp_in_images)
"""

#=======================================================

if (PROVIDER == 'inpe'):
  products = api.query(location=extent,
                     initial_date=date(int(START_DATE.split('-')[0]), int(START_DATE.split('-')[1]), int(START_DATE.split('-')[2])),
                     end_date=date(int(END_DATE.split('-')[0]), int(END_DATE.split('-')[1]), int(END_DATE.split('-')[2])),
                     cloud=25, # 0 = no cloud - 100 = can have any cloud
                     limit=100,
                     collections=[product_type])
  print('products: ', len(products['features']))

#=======================================================

fullres_model = load_model(fp_in_model + 'u-net-model-with-submission/fullres_model.h5')

#=======================================================

def predict(img, image_dir='/content/tiles/rgb/'):
    c_img = imread(os.path.join(image_dir, img))
    c_img = np.expand_dims(c_img, 0) / 255.0
    cur_seg = fullres_model.predict(c_img)[0]
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

def pred_encode(img, **kwargs):
    cur_seg, _ = predict(img)
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

def cut_tif_into_tiles(fp_band_blue, fp_band_green, fp_band_red, fp_band_nir):
  !rm -rf /content/tiles/blue
  !rm -rf /content/tiles/green
  !rm -rf /content/tiles/red
  !rm -rf /content/tiles/nir

  fp_tiles = '/content/tiles/'
  fp_blue = '/content/tiles/blue/'
  fp_green = '/content/tiles/green/'
  fp_red = '/content/tiles/red/'
  fp_nir = '/content/tiles/nir/'

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

def split_and_resize_images(split):
  fp_rgb = '/content/tiles/rgb/'
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
      case '3x3':
        # Resize
        resized = np.array(Image.fromarray((image * 255).astype(np.uint8)).resize((768, 768)))
        if '.jpg' in img_path: resized = cv2.bitwise_not(resized)

        # Save
        !rm $img_path
        imsave(img_path, resized)

      case '6x6':
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
        !rm $img_path

#=======================================================

def predict_ships():
  out_pred_rows = []
  for img_name in os.listdir('/content/tiles/rgb/'):
    if 'ndwi' not in img_name:
      print("Predicting ships in image ", img_name)
      if (img_name == '.ipynb_checkpoints'): pass
      else: out_pred_rows += pred_encode(img_name, min_max_threshold=1.0)

  print(out_pred_rows)
  if (out_pred_rows != []):
    sub = pd.DataFrame(out_pred_rows)
    sub.columns = ['ImageId', 'EncodedPixels']
    sub = sub[sub.EncodedPixels.notnull()]
    if (len(sub.index) == 1): sub = pd.concat([sub,sub.loc[[0],:]]).sort_index() # Make copy if single row
    return sub
  else:
    return None

#=======================================================

def plot_predictions(sub, file_name, split):
  print('Total ships in all images: ', len(sub.index))
  TOP_PREDICTIONS = len(sub.index)
  if (TOP_PREDICTIONS > 30): TOP_PREDICTIONS = 30
  fig, m_axs = plt.subplots(TOP_PREDICTIONS, 3, figsize = (14, TOP_PREDICTIONS*4))
  [c_ax.axis('off') for c_ax in m_axs.flatten()]

  def analyse_predictions(img, array_of_ships, prop):
    img_path = '/content/tiles/rgb/'
    img_ndwi = img.replace('RGB', 'ndwi')
    img_ndwi = img_ndwi.replace('.jpg', '.tif')

    band_ndwi = rasterio.open(img_path + img_ndwi)
    ndwi = band_ndwi.read(1)
    ndwi_base = np.copy(ndwi)
    ndwi[ndwi_base <= 0.0] = 0 # = land
    ndwi[ndwi_base > 0.0] = 1 # = water

    # for x in range(prop.bbox[1], prop.bbox[3]):
    #   if x < 768 and prop.bbox[0] < 768:
    #     if ndwi[x][prop.bbox[0]] == 0: return False
    # for x in range(prop.bbox[1], prop.bbox[3]):
    #   if x < 768 and prop.bbox[2] < 768:
    #     if ndwi[x][prop.bbox[2]] == 0: return False
    # for y in range(prop.bbox[0], prop.bbox[2]):
    #   if y < 768 and prop.bbox[1] < 768:
    #     if ndwi[prop.bbox[1]][y] == 0: return False
    # for y in range(prop.bbox[0], prop.bbox[2]):
    #   if y < 768 and prop.bbox[3] < 768:
    #     if ndwi[prop.bbox[3]][y] == 0: return False

    # x1, x2, y1, y2 = prop.bbox[1], prop.bbox[3], prop.bbox[0], prop.bbox[2]
    # if x1 >= 768: x1 = 767
    # if x2 >= 768: x2 = 767
    # if y1 >= 768: y1 = 767
    # if y2 >= 768: y2 = 767
    # midx = (x1 + x2)//2
    # midy = (y1 + y2)//2
    # if ndwi[midx][y1] == 0: return False
    # if ndwi[midx][y2] == 0: return False
    # if ndwi[x1][midy] == 0: return False
    # if ndwi[x2][midy] == 0: return False

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

  def raw_prediction(img, image_dir='/content/tiles/rgb/'):
      c_img = imread(os.path.join(image_dir, img))
      c_img = np.expand_dims(c_img, 0)/255.0
      cur_seg = fullres_model.predict(c_img)[0]
      return cur_seg, c_img[0]

  for (ax1, ax2, ax3), img in zip(m_axs, sub.ImageId.unique()[:TOP_PREDICTIONS]):
      pred, c_img = raw_prediction(img)
      print("Img: ", img)
      array_of_ships = masks_as_color(sub.query('ImageId==\"{}\"'.format(img))['EncodedPixels'])

      lbl_0 = label(pred[...,0])
      props = regionprops(lbl_0)
      img_1 = c_img.copy()
      for prop in props:
          print('Found bbox ', prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2])

          if analyse_predictions(img, array_of_ships, prop):
            cv2.rectangle(img_1, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (0, 255, 0), 2)
          else:
            cv2.rectangle(img_1, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)

      #if ('220' in img): return pred, c_img, array_of_ships

      ax1.imshow(c_img)
      ax1.set_title('Image: ' + img)
      ax2.imshow(np.sum(np.transpose(array_of_ships, (1, 2, 0)), 2), cmap=get_cmap('jet'))
      ax2.set_title('Prediction')
      ax3.imshow(img_1)
      ax3.set_title('Masks')

      fp_output = '/content/output/'
      if not os.path.isdir(fp_output): os.mkdir(fp_output)
      fig.savefig(fp_output + file_name + '_' + split + '.jpg', dpi=200, bbox_inches='tight', pad_inches=0)
      plt.close('all')

#=======================================================

def plot_ndwi_predictions(sub, file_name, split):
  print('Total ships in all images: ', len(sub.index))
  TOP_PREDICTIONS = len(sub.index)
  num_of_samples = 4
  if (TOP_PREDICTIONS > num_of_samples): TOP_PREDICTIONS = num_of_samples # = num_of_samples
  fig, m_axs = plt.subplots(TOP_PREDICTIONS, 4, figsize = (16, TOP_PREDICTIONS*4))
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

  def raw_prediction(img, image_dir='/content/tiles/rgb/'):
      c_img = imread(os.path.join(image_dir, img))
      c_img = np.expand_dims(c_img, 0)/255.0
      cur_seg = fullres_model.predict(c_img)[0]
      return cur_seg, c_img[0]

  first = True
  for (ax1, ax2, ax3, ax4), img in zip(m_axs, sub[sub['ImageId'].isin(['tile_RGB_201.jpg',
                                                                       'tile_RGB_255.jpg',
                                                                       'tile_RGB_184.jpg',
                                                                       'tile_RGB_294.jpg'])].ImageId.unique()): #[:TOP_PREDICTIONS]
      # Get prediction
      pred, c_img = raw_prediction(img)
      print("Img: ", img)
      array_of_ships = masks_as_color(sub.query('ImageId==\"{}\"'.format(img))['EncodedPixels'])

      # Get ndwi
      img_path = '/content/tiles/rgb/'
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
      ax2.imshow(ndwi)
      if first: ax2.set_title('Water mask', fontsize=18)
      ax3.imshow(np.sum(np.transpose(array_of_ships, (1, 2, 0)), 2), cmap=get_cmap('jet'))
      if first: ax3.set_title('Ship mask', fontsize=18)
      ax4.imshow(img_1)
      if first: ax4.set_title('Bounding boxes', fontsize=18)
      first = False

      fp_output = '/content/output/'
      if not os.path.isdir(fp_output): os.mkdir(fp_output)
      fig.savefig(fp_output + file_name + '_' + split + '.jpg', dpi=200, bbox_inches='tight', pad_inches=0)
      plt.close('all')

#=======================================================

def plot_resized_predictions(sub, file_name, split):
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

  def raw_prediction(img, image_dir='/content/tiles/rgb/'):
      c_img = imread(os.path.join(image_dir, img))
      c_img = np.expand_dims(c_img, 0)/255.0
      cur_seg = fullres_model.predict(c_img)[0]
      return cur_seg, c_img[0]

  first = True
  count_combined = 0
  combined_imgs = np.zeros(num_of_samples).astype(int)
  img_array = np.zeros((4, 768, 768, 3))
  ship_array = np.zeros((4, 768, 768))
  boxes_array = np.zeros((4, 768, 768, 3))
  combined_img_array = np.zeros((4, 1536, 1536, 3))
  combined_ship_array = np.zeros((4, 1536, 1536))
  combined_boxes_array = np.zeros((4, 1536, 1536, 3))
  for img_block in [['tile_RGB_273_1.jpg',
                    'tile_RGB_273_2.jpg',
                    'tile_RGB_273_3.jpg',
                    'tile_RGB_273_4.jpg'],
                    ['tile_RGB_292_1.jpg',
                    'tile_RGB_292_2.jpg',
                    'tile_RGB_292_3.jpg',
                    'tile_RGB_292_4.jpg'],
                    ['tile_RGB_274_1.jpg',
                    'tile_RGB_274_2.jpg',
                    'tile_RGB_274_3.jpg',
                    'tile_RGB_274_4.jpg'],
                    ['tile_RGB_275_1.jpg',
                    'tile_RGB_275_2.jpg',
                    'tile_RGB_275_3.jpg',
                    'tile_RGB_275_4.jpg']]:
    count_block = 0
    for img in img_block:
      # Get prediction
      pred, c_img = raw_prediction(img)
      print("Img: ", img)
      array_of_ships = masks_as_color(sub.query('ImageId==\"{}\"'.format(img))['EncodedPixels'])

      # Get ndwi
      img_path = '/content/tiles/rgb/'
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
      img_boxes = c_img.copy()
      for prop in props:
          print('Found bbox ', prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2])

          if analyse_predictions(img, array_of_ships, prop, ndwi):
            cv2.rectangle(img_boxes, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (0, 255, 0), 2)
          else:
            cv2.rectangle(img_boxes, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)

      img_array[count_block] = c_img
      ship_array[count_block] = np.sum(np.transpose(array_of_ships, (1, 2, 0)), 2)
      boxes_array[count_block] = img_boxes
      count_block += 1

    top = np.concatenate((img_array[0], img_array[1]),axis=1)
    btm = np.concatenate((img_array[2], img_array[3]),axis=1)
    combined = np.concatenate((top, btm), axis=0)
    combined_img_array[count_combined] = combined

    top = np.concatenate((ship_array[0], ship_array[1]),axis=1)
    btm = np.concatenate((ship_array[2], ship_array[3]),axis=1)
    combined = np.concatenate((top, btm), axis=0)
    combined_ship_array[count_combined] = combined

    top = np.concatenate((boxes_array[0], boxes_array[1]),axis=1)
    btm = np.concatenate((boxes_array[2], boxes_array[3]),axis=1)
    combined = np.concatenate((top, btm), axis=0)
    combined_boxes_array[count_combined] = combined

    combined_imgs[count_combined] = count_combined
    count_combined += 1

  for (ax1, ax2, ax3), img in zip(m_axs, combined_imgs):
    ax1.imshow(combined_img_array[img])
    if first: ax1.set_title('Image', fontsize=18)#('Image: ' + img, fontsize=15)
    ax2.imshow(combined_ship_array[img], cmap=get_cmap('jet'))
    if first: ax2.set_title('Ship mask', fontsize=18)
    ax3.imshow(combined_boxes_array[img])
    if first: ax3.set_title('Bounding boxes', fontsize=18)
    first = False

    fp_output = '/content/output/'
    if not os.path.isdir(fp_output): os.mkdir(fp_output)
    fig.savefig(fp_output + file_name + '_' + split + '.jpg', dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close('all')

#=======================================================

def plot_sized_predictions(sub, file_name, split):
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

  def raw_prediction(img, image_dir='/content/tiles/rgb/'):
      c_img = imread(os.path.join(image_dir, img))
      c_img = np.expand_dims(c_img, 0)/255.0
      cur_seg = fullres_model.predict(c_img)[0]
      return cur_seg, c_img[0]

  first = True
  for (ax1, ax2, ax3), img in zip(m_axs, np.flip(sub[sub['ImageId'].isin(['tile_RGB_273.jpg',
                                                                  'tile_RGB_292.jpg',
                                                                  'tile_RGB_274.jpg',
                                                                  'tile_RGB_275.jpg'])].ImageId.unique())): #[:TOP_PREDICTIONS]
      # Get prediction
      pred, c_img = raw_prediction(img)
      print("Img: ", img)
      array_of_ships = masks_as_color(sub.query('ImageId==\"{}\"'.format(img))['EncodedPixels'])

      # Get ndwi
      img_path = '/content/tiles/rgb/'
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

      fp_output = '/content/output/'
      if not os.path.isdir(fp_output): os.mkdir(fp_output)
      fig.savefig(fp_output + file_name + '_' + split + '.jpg', dpi=200, bbox_inches='tight', pad_inches=0)
      plt.close('all')

#=======================================================


