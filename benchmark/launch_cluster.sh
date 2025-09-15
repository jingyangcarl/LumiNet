interactive_job_8gpu() {
  srun -A ai4m_video --partition interactive,grizzly,polar,polar3,polar4,batch_singlenode \
      --container-mounts=$HOME:/home,/lustre:/lustre \
      --exclusive --gpus=8 --ntasks-per-node=8 --cpus-per-task=16 -t 3:59:00 \
      --pty bash -c "bash && cd /lustre/fsw/portfolios/maxine/users/jingya/projects/LumiNet"
}

interactive_job_8gpu 
