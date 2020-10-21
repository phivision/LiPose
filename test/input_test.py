import scipy.io as sio
import imageio
import pylab
import pathlib

DIR_PATH = "/home/fanghao/Downloads/SURREAL_v1/cmu/train/run0/ung_90_10"
FILE_NAME = 'ung_90_10_c0001'
NUM_FRAMES = 10

if __name__ == "__main__":
    dir_path = pathlib.Path(DIR_PATH)
    mp4_file = dir_path.joinpath(FILE_NAME + '.mp4')
    vid = imageio.get_reader(mp4_file,  'ffmpeg')
    meta = vid.get_meta_data()

    depth_file = dir_path.joinpath(FILE_NAME + '_depth.mat')
    with depth_file.open(mode='r'):
        depth_data = sio.loadmat(depth_file)

    info_file = dir_path.joinpath(FILE_NAME + '_info.mat')
    with info_file.open(mode='r'):
        info_data = sio.loadmat(info_file)

    num_frames = meta['nframes'] if meta['nframes'] <= NUM_FRAMES else NUM_FRAMES
    for num in range(num_frames):
        image = vid.get_data(num)
        fig = pylab.figure()
        fig.suptitle('image #{}'.format(num), fontsize=20)
        pylab.imshow(image)
        dmap = depth_data['depth_'+str(num+1)]
        fig = pylab.figure()
        fig.suptitle('depth #{}'.format(num), fontsize=20)
        pylab.imshow(dmap)
    pylab.show()


