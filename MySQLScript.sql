CREATE DATABASE Trading;
USE Trading;
CREATE USER 'Filler'@'localhost' IDENTIFIED BY '5gs.1-';
GRANT INSERT ON Trading . * TO 'Filler'@'localhost';
CREATE TABLE Gold(Timestamp BIGINT, Close FLOAT, Volume INTEGER);

