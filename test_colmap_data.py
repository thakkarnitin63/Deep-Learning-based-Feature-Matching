from features.colmap_database_reader import COLMAPDatabaseReader


def main():
    # Provide the path to your COLMAP SQLite database file
    db_file_path = "colmap_run/buddha.db"

    # Create an instance of COLMAPDatabaseReader
    db_reader = COLMAPDatabaseReader(db_file_path)

    # Connect to the database
    db_reader.connect()

    # Fetch and print some relevant data
    keypoints = db_reader.fetch_all_keypoints()
    print(
        f"Number of keypoints:: {len(keypoints)},  Keypoints datatype :: {type(keypoints[0])}"
    )
    descriptors = db_reader.fetch_all_descriptors()
    print(
        f"Number of descriptor:: {len(descriptors)},  Descriptors datatype :: {type(descriptors[0])}"
    )
    cameras = db_reader.fetch_all_cameras()
    print(
        f"Number of cameras:: {len(cameras)},  cameras datatype :: {type(cameras[0])}"
    )

    matches = db_reader.fetch_all_matches()
    print("Number of matches:", len(matches))
    two_view_geometries = db_reader.fetch_all_two_view_geometries()
    print("Number of two-view geometries:", len(two_view_geometries))

    # Close the database connection
    db_reader.close()


if __name__ == "__main__":
    main()
