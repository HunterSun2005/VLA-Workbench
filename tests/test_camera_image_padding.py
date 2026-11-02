"""Geometry checks use synthetic RGB frames, without device or model access."""
import numpy as np
import pytest

from vla.core.images import resize_rgb_with_pad


@pytest.mark.parametrize('source,target,content', [
    ((480,640),(224,224),(168,224)),
    ((640,480),(224,224),(224,168)),
    ((301,503),(128,255),(128,214)),
    ((16,16),(32,32),(32,32)),
    ((224,224),(224,224),(224,224)),
])
def test_letterbox_retains_all_four_edges(source, target, content):
    h,w=source
    image=np.full((h,w,3),100,dtype=np.uint8)
    image[:,:w//8]=[255,0,0]
    image[:,-w//8:]=[0,255,0]
    image[:h//8]=[0,0,255]
    image[-h//8:]=[255,255,0]
    before=image.copy()
    result=resize_rgb_with_pad(image,*target)
    ch,cw=content
    top,left=(target[0]-ch)//2,(target[1]-cw)//2
    assert result.shape==(*target,3) and result.dtype==np.uint8 and result.flags.c_contiguous
    np.testing.assert_array_equal(result[top+ch//2,left],[255,0,0])
    np.testing.assert_array_equal(result[top+ch//2,left+cw-1],[0,255,0])
    np.testing.assert_array_equal(result[top,left+cw//2],[0,0,255])
    np.testing.assert_array_equal(result[top+ch-1,left+cw//2],[255,255,0])
    assert not result[:top].any() and not result[top+ch:].any()
    assert not result[:,:left].any() and not result[:,left+cw:].any()
    np.testing.assert_array_equal(image,before)
    assert not np.shares_memory(result,image)
    if source==target:np.testing.assert_array_equal(result,image)


@pytest.mark.parametrize('image,target', [
    (np.zeros((0,10,3),dtype=np.uint8),(224,224)),
    (np.zeros((10,10),dtype=np.uint8),(224,224)),
    (np.zeros((10,10,3),dtype=np.float32),(224,224)),
    (np.zeros((10,10,3),dtype=np.uint8),(0,224)),
])
def test_letterbox_rejects_invalid_frames_and_size(image,target):
    with pytest.raises(ValueError):resize_rgb_with_pad(image,*target)
