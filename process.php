<?php
/*
 @Author: Sanjay Kumar
 @Project: PHP MySQLi File Upload with Progress Bar Using jQuery Ajax
 @Email: onlinewebtutorhub@gmail.com
 @Website: https://onlinewebtutorblog.com/
*/
// Include the database configuration file 
require 'dbconfig.php';

if (isset($_POST['btnSubmit'])) {

    if (preg_match("/\S+/", $_FILES["uploadImage"]["tmp_name"]) === 0) {
        echo "*The image field is required.";
        exit();
    }

    $uploadfile = $_FILES["uploadImage"]["tmp_name"];
    $folderPath = "uploads/";

    if (!is_writable($folderPath) || !is_dir($folderPath)) {
        echo "Please check folder permission";
        exit();
    }
    if (move_uploaded_file($_FILES["uploadImage"]["tmp_name"], $folderPath . $_FILES["uploadImage"]["name"])) {

        $stmt = $conn->prepare("INSERT INTO images (image_name) VALUES (?)");
        $stmt->bind_param("s", $filename);
        $filename = $folderPath . $_FILES["uploadImage"]["name"];

        $stmt->execute();

        echo '<img src="' . $folderPath . "" . $_FILES["uploadImage"]["name"] . '">';
        exit();
    }
}
