#!/bin/sh

setenforce 0 &&
chmod 777 /dev/ttyACM0 &&
chown u0_a109 /dev/ttyACM0
