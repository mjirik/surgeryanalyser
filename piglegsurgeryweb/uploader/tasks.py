from .models import UploadedFile

def run_processing(serverfile:UploadedFile):
    log_format = loguru._defaults.LOGURU_FORMAT
    logger_id = logger.add(
        str(Path(serverfile.outputdir) / "log.txt"),
        format=log_format,
        level='DEBUG',
        rotation="1 week",
        backtrace=True,
        diagnose=True
    )
    delete_generated_images(serverfile) # remove images from database and the output directory
    mainapp = scaffan.algorithm.Scaffan()
    mainapp.set_input_file(serverfile.imagefile.path)
    mainapp.set_output_dir(serverfile.outputdir)
    fn,_,_ = models.get_common_spreadsheet_file(serverfile.owner)
    mainapp.set_common_spreadsheet_file(str(fn).replace("\\", "/"))
    # settings.SECRET_KEY
    logger.debug("Scaffan processing run")
    if len(centers_mm) > 0:
        mainapp.set_parameter("Input;Lobulus Selection Method", "Manual")
    else:
        mainapp.set_parameter("Input;Lobulus Selection Method", "Auto")
    mainapp.run_lobuluses(seeds_mm=centers_mm)
    serverfile.score = _clamp(mainapp.report.df["SNI area prediction"].mean() * 0.5, 0., 1.)
    serverfile.score_skeleton_length = mainapp.report.df["Skeleton length"].mean()
    serverfile.score_branch_number = mainapp.report.df["Branch number"].mean()
    serverfile.score_dead_ends_number = mainapp.report.df["Dead ends number"].mean()
    serverfile.score_area = mainapp.report.df["Area"].mean()

    add_generated_images(serverfile) # add generated images to database

    serverfile.processed_in_version = scaffan.__version__
    serverfile.process_started = False
    serverfile.last_error_message = ''
    if serverfile.zip_file and Path(serverfile.zip_file.path).exists():
        serverfile.zip_file.delete()

    views.make_zip(serverfile)
    serverfile.save()
    logger.remove(logger_id)

