@echo off
set long_delay=40
set mid_delay=20
set short_delay=2

:: ---- START SERVICES ----
echo Starting Hadoop DFS...
cd C:\Hadoop\hadoop-3.4.0\sbin
call start-dfs.cmd
timeout /t %long_delay% /nobreak

echo Leaving HDFS safe mode...
call hdfs dfsadmin -safemode leave
timeout /t %short_delay% /nobreak

echo Starting HBase...
cd C:\HBase\hbase-2.5.11\bin
call start-hbase.cmd
timeout /t %mid_delay% /nobreak

echo Starting HBase Thrift in a new window...
start "HBaseThriftWindow" cmd /k "hbase thrift start"
timeout /t %mid_delay% /nobreak

echo All services started.
echo Press any key in this window to close all service windows.
pause

:: ---- STOP SERVICES GRACEFULLY ----
echo Stopping HBase Thrift...
call hbase thrift stop
timeout /t %short_delay% /nobreak

echo Stopping HBase...
call stop-hbase.cmd
timeout /t %short_delay% /nobreak

echo Stopping Hadoop DFS...
cd C:\Hadoop\hadoop-3.4.0\sbin
call stop-dfs.cmd
timeout /t %short_delay% /nobreak

echo Graceful shutdown complete.

:: ---- FORCE CLOSE REMAINING WINDOWS ----
taskkill /F /T /FI "WINDOWTITLE eq HBaseThriftWindow - hbase  thrift start"
taskkill /F /T /FI "WINDOWTITLE eq HBase Distribution - C:\HBase\hbase-2.5.11\bin\hbase.cmd   master start"
taskkill /F /T /FI "WINDOWTITLE eq Apache Hadoop Distribution - hadoop   namenode"
taskkill /F /T /FI "WINDOWTITLE eq Apache Hadoop Distribution - hadoop   datanode"

echo All service windows closed.
pause
