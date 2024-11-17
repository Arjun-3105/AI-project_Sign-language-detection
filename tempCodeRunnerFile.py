import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from flask import Flask, render_template, Response, request, jsonify