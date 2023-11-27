from mmpose.apis import inference

if __name__ == '__main__':

    config='D:\\工作和学习文档\\IIT\\CS577\\project\\ViTPose-main\\configs\\body\\2d_kpt_sview_rgb_img\\topdown_heatmap\\coco\\ViTPose_small_simple_coco_256x192.py'
    checkPoint='D:\\工作和学习文档\\IIT\\CS577\\project\\vitpose\\vitpose_small_up4.pth'
    img_path='D:\\工作和学习文档\\IIT\\CS577\\project\\vitpose\\000029.jpg'

    model = inference.init_pose_model(config,checkPoint)
    inference.inference_top_down_pose_model(model,img_path,format='xywh',dataset='TopDownCocoDataset')