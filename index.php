<?php
/*
 @Author: Sanjay Kumar
 @Project: PHP MySQLi File Upload with Progress Bar Using jQuery Ajax
 @Email: onlinewebtutorhub@gmail.com
 @Website: https://onlinewebtutorblog.com/
*/
?>

<!DOCTYPE html>
<html>

<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>uploader</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.1/dist/css/bootstrap.min.css" />
    <link rel="stylesheet" href="style.css" />
</head>

<body>
    <div class="container mt-5">
        <div class="row">
            <div class="col-md-12">
                <div class="card w-75 m-auto">
                    <div class="card-header text-center bg-primary text-white">
                        <h4>uploaderprocess</h4>
                    </div>
                    <div class="card-body">
                        <div class='progress' id="progressDivId">
                            <div class='progress-bar' id='progressBar'></div>
                            <div class='percent' id='percent'>0%</div>
                        </div>
                        <div id='outputImage'></div>
                        <div id='error'></div>
                        <form action="javascript:void(0)" id="uploadForm" name="frmupload" method="post" enctype="multipart/form-data">
                            <div class="form-group">
                                <label><strong>Select Image : </strong><span class="text-danger"> *</span></label>
                                <input type="file" id="uploadImage" name="uploadImage" class="form-control">
                            </div>
                            <br>
                            <div class="d-flex">
                                <input id="submitButton" type="submit" name='btnSubmit' value="Upload" class="btn btn-success">
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://code.jquery.com/jquery-3.3.1.min.js" type="text/javascript"></script>
    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jquery.form/4.3.0/jquery.form.min.js"></script>
    <script type="text/javascript" src="script.js"></script>

</body>

</html>