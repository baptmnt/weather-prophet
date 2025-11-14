import os
if __name__ == "__main__":
    commands = [
        #"py tests-louis/create_ml_dataset.py --data-root Z:/downloads/dataset/data/ --output-dir dataset/ --zone SE --year 2016  --station-id 69029001 --num-workers 0 --use-dask --downsample-factor 1 --sample-interval 3600",
        "py tests-louis/create_ml_dataset.py --data-root Z:/downloads/dataset/data/ --output-dir dataset/ --zone SE --year 2016  --station-id 69029001 --num-workers 0 --use-dask --downsample-factor 5 --sample-interval 3600",
        #"py tests-louis/create_ml_dataset.py --data-root Z:/downloads/dataset/data/ --output-dir dataset/ --zone SE --year 2016  --station-id 69029001 --num-workers 0 --use-dask --downsample-factor 10 --sample-interval 3600",
        #"py tests-louis/create_ml_dataset.py --data-root Z:/downloads/dataset/data/ --output-dir dataset/ --zone SE --year 2017  --station-id 69029001 --num-workers 0 --use-dask --downsample-factor 1 --sample-interval 3600",
        #"py tests-louis/create_ml_dataset.py --data-root Z:/downloads/dataset/data/ --output-dir dataset/ --zone SE --year 2017  --station-id 69029001 --num-workers 0 --use-dask --downsample-factor 2 --sample-interval 3600",
        #"py tests-louis/create_ml_dataset.py --data-root Z:/downloads/dataset/data/ --output-dir dataset/ --zone SE --year 2017  --station-id 69029001 --num-workers 0 --use-dask --downsample-factor 5 --sample-interval 3600",
        #"py tests-louis/create_ml_dataset.py --data-root Z:/downloads/dataset/data/ --output-dir dataset/ --zone SE --year 2017  --station-id 69029001 --num-workers 0 --use-dask --downsample-factor 10 --sample-interval 3600",
        #"py tests-louis/create_ml_dataset.py --data-root Z:/downloads/dataset/data/ --output-dir dataset/ --zone SE --year 2018  --station-id 69029001 --num-workers 0 --use-dask --downsample-factor 1 --sample-interval 3600",
        #"py tests-louis/create_ml_dataset.py --data-root Z:/downloads/dataset/data/ --output-dir dataset/ --zone SE --year 2018  --station-id 69029001 --num-workers 0 --use-dask --downsample-factor 2 --sample-interval 3600",
        #"py tests-louis/create_ml_dataset.py --data-root Z:/downloads/dataset/data/ --output-dir dataset/ --zone SE --year 2018  --station-id 69029001 --num-workers 0 --use-dask --downsample-factor 5 --sample-interval 3600",
        #"py tests-louis/create_ml_dataset.py --data-root Z:/downloads/dataset/data/ --output-dir dataset/ --zone SE --year 2018  --station-id 69029001 --num-workers 0 --use-dask --downsample-factor 10 --sample-interval 3600",


    ]

    for command in commands:
        try:
            print(f"Executing: {command}")
            os.system(command)
        except Exception as e:
            print(f"Error executing {command}: {e}")
            continue
        finally:
            print(f"Finished: {command}")
    
