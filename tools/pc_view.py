import pcl
import pcl.pcl_visualization


def view(pc_path):
    
    visual = pcl.pcl_visualization.CloudViewing()
    

    v = True
    while v:
        try:
            cloud = pcl.load_XYZRGB(pc_path)
            if len(cloud.to_list()) == 0:
                continue
            visual.ShowColorCloud(cloud)
        except IndexError as e:
            continue
        
        v = not(visual.WasStopped())


if __name__ == "__main__":
    pc_path = '../test/pc/segmented_cloud.pcd'
    view(pc_path)