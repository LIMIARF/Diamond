# Please train the model using singularity with the following command
timeout 24h python main.py --submission_name Submission_1 --weights_pth_file ./backbone/resnet101.pth

# # Second run with a 2-day limit. Convert days to hours: 2 days * 24 hours/day = 48 hours
# timeout 48h python main.py --learning_rate 1e-3 --batch_size 8 --backbone efficientnet_b0

# # Third run with a 2-day limit. Convert days to hours: 2 days * 24 hours/day = 48 hours
# timeout 48h python main.py --learning_rate 1e-3 --batch_size 8 --backbone resnet101