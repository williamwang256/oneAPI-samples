# `Host Pipes` Sample

This FPGA sample is a tutorial that demonstrates how to use pipes to send and receive data between a host and a device.

| Area                  | Description
|:--                    |:--
| What you will learn   | Basics of host pipe declaration and usage
| Time to complete      | 30 minutes
| Category              | Concepts and Functionality

## Purpose

Pipes are a first-in first-out (FIFO) buffer construct that provides links between elements of a design. Access pipes through read and write application programming interfaces (APIs), without the notion of a memory address or pointer to elements within the FIFO.

Pipes connecting a host and a device are called host pipes. Use host pipes to move data between the host part of a design and a kernel that resides on the FPGA. A read and write API imposes FIFO ordering on accesses to this data. The advantage to this approach is that you do not need to write code to address specific locations in these buffers when accessing the data. Host pipes provide a "streaming" interface between host and FPGA, and are best used in designs where random access to data is not needed or wanted.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware             | Intel® Agilex® 7, Arria® 10, and Stratix® 10 FPGAs
| Software             | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.

> **Warning** Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

This sample is part of the FPGA code samples. It is categorized as a Tier 2 sample that demonstrates a compiler feature.

```mermaid
flowchart LR
   tier1("Tier 1: Get Started")
   tier2("Tier 2: Explore the Fundamentals")
   tier3("Tier 3: Explore the Advanced Techniques")
   tier4("Tier 4: Explore the Reference Designs")

   tier1 --> tier2 --> tier3 --> tier4

   style tier1 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier2 fill:#f96,stroke:#333,stroke-width:1px,color:#fff
   style tier3 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier4 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
```

Find more information about how to navigate this part of the code samples in the [FPGA top-level README.md](/DirectProgramming/C++SYCL_FPGA/README.md).
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), and more.


## Key Implementation Details

This tutorial illustrates some key concepts:

- Declaring a Host Pipe
- Host Pipe Read/Write API

### Declaring a Host Pipe

Each individual host pipe is a function scope class declaration of the templated `pipe` class. The first template parameter should be a user-defined type that differentiates this particular pipe from the others. The second template parameter defines the datatype of each element carried by the pipe. The third template parameter defines the pipe capacity, which is the guaranteed minimum number of elements of datatype that can be held in the pipe. In other words, for a given pipe with capacity `c`, the compiler guarantees that operations on the pipe will not block due to capacity as long as, for any consecutive `n` operations on the pipe, the number of writes to the pipe minus the number of reads does not exceed `c`. These template parameters are summarized below, and must be specified in the given order.

| Template Parameter | Valid Values | Default Values           |
| -------------------| ------------ | ------------------------ |
| `id`                          | type         | none (must be specified) |
| `type`                        | type         | none (must be specified) |
| `min_capacity` (optional)     | integer ≥ 0  | 0                        |
| `properties` (optional)       | see below    | see below                |

The fourth template parameter uses the oneAPI properties class to allow users to define additional semantic properties for a host pipe. These are summarized in the table below, and can be specified in *any* order. Omitting a single property from the properties class instructs the compiler to assume the default value for that property (i.e., you can just define the properties you would like to change from the default). Omitting the properties template parameter entirely instructs the compiler to assume the default values for *all* these properties.

| Property | Valid Values | Default Values |
| ---------| ------------ | -------------- |
| `ready_latency<int>`                    | integer ≥ 0 | 0 |
| `bits_per_symbol<int>`                  | integer ≥ 0 that divides the size of the data type | 8 |
| `uses_valid<bool>`                      | boolean      | `true` |
| `first_symbol_in_high_order_bits<bool>` | boolean      | `true` |
| `protocol`                              | `protocol_avalon_streaming` / `protocol_avalon_streaming_uses_ready` / `protocol_avalon_mm` / `protocol_avalon_mm_uses_ready` | `protocol_avalon_streaming_uses_ready` |

For more information on the definitions and usage of these properties, see the [FPGA Optimization Guide for Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/docs/oneapi-fpga-add-on/optimization-guide/current/host-pipe-declaration.html).

In the following example, `FirstPipeT` and `SecondPipeT` are unique user-defined types that identify two host pipes. The first host pipe (which has been aliased to `FirstPipeInstance`), carries `int` type data elements and has a capacity of `8`. The second host pipe (`SecondPipeInstance`) carries `float` type data elements, and has a capacity of `8`. Using aliases allows these pipes to be referred to by a shorter and more descriptive handle, rather than having to repeatedly type out the full namespace and template parameters.

```c++
// Unique user-defined types
class FirstPipeT;
class SecondPipeT;

// Host pipe properties (listed here are the defaults; this achieves the same
// behavior as not specifying any of these properties)
using PipePropertiesT = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::ready_latency<0>,
    sycl::ext::intel::experimental::bits_per_symbol<8>,
    sycl::ext::intel::experimental::uses_valid<true>,
    sycl::ext::intel::experimental::first_symbol_in_high_order_bits<true>),
    sycl::ext::intel::experimental::protocol_avalon_streaming_uses_ready
);

using FirstPipeInstance = sycl::ext::intel::experimental::pipe<
    FirstPipeT,      // An identifier for the pipe
    int,             // The type of data in the pipe
    8,               // The capacity of the pipe
    PipePropertiesT  // Customizable pipe properties
    >;
    
using SecondPipeInstance = sycl::ext::intel::experimental::pipe<
    SecondPipeT,     // An identifier for the pipe
    float,           // The type of data in the pipe
    4                // The capacity of the pipe
    PipePropertiesT  // Customizable pipe properties
    >;
```

### Avalon Streaming Sideband Signals

You can enable Avalon streaming sideband signal support by using the special `StreamingBeat` struct provided by the `pipes_ext.hpp` header file. Only the `StreamingBeat` struct generates sideband signals when used with a host pipe.

The `StreamingBeat` struct is templated on three parameters. The first template parameter controls the data type. The second parameter is used to enable additional 1-bit `start_of_packet` (`sop`) and `end_of_packet` (`eop`) signals to the Avalon interface. The third and final template parameter is used to enable the `empty` signal, which indicates the number of symbols that are empty during the `eop` cycle.

The following example uses the `StreamingBeat` struct with the `sop`, `eop`, and `empty` signals added (the second and third template parameters are set to `true`). 

```c++
#include <sycl/ext/intel/prototype/pipes_ext.hpp>

...

using PipeDataT = ac_int<kBitsPerSymbol * kSymbolsPerBeat, false>;
using StreamingBeatDataT = sycl::ext::intel::experimental::StreamingBeat<PipeData, true, true>;

// Host pipe properties
using PipePropertiesT = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::bits_per_symbol<kBitsPerSymbol>,
    sycl::ext::intel::experimental::protocol_avalon_streaming_uses_ready
);

using ThirdPipeInstance = sycl::ext::intel::experimental::pipe<
    ThirdPipeT,         // An identifier for the pipe
    StreamingBeatData,  // The type of data in the pipe
    8,                  // The capacity of the pipe
    PipePropertiesT     // Customizable pipe properties
    >;
```

> **Note**: The size of the `PipeDataT` type must be a multiple of `bits_per_symbol`, which is specified in the associated host pipe declaration.

See the [FPGA Optimization Guide for Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/docs/oneapi-fpga-add-on/optimization-guide/current/host-pipes-rtl-interfaces.html) for more information on configuring RTL interfaces.

### Host Pipe API

Host Pipes expose read and write interfaces that allow a single element to be read or written in FIFO order to the pipe. These read and write interfaces are static class methods on the templated classes described in the [Declaring a Host Pipe](#declaring-a-host-pipe) section. The API provides the following interfaces:

  - [Blocking write interface](#blocking-write)
  - [Non-blocking write interface](#non-blocking-write)
  - [Blocking read interface](#blocking-read)
  - [Non-blocking read interface](#non-blocking-read)

#### Blocking Write

The host pipe write interface writes a single element of the given datatype (`int` in the examples below) to the host pipe. On the host side, this class method takes a SYCL* device queue argument as its first argument, and the element being written as its second argument.

```c++
queue q(...);
...
int data_element = ...;

// blocking write from host to pipe
FirstPipeInstance::write(q, data_element);
```

In the FPGA kernel, writes to a host pipe take a single argument, which is the element being written.

```c++
float data_element = ...;

// blocking write from device to pipe
SecondPipeInstance::write(data_element);
```

#### Non-Blocking Write

Non-blocking writes add a `bool` argument in both host and device APIs that is passed by reference and returns true in this argument if the write was successful, and false if it was unsuccessful.

On the host:

```c++
queue q(...);
...
int data_element = ...;

// variable to hold write success or failure
bool success = false;

// attempt non-blocking write from host to pipe until successful
while (!success) FirstPipeInstance::write(q, data_element, success);
```

On the device:

```c++
float data_element = ...;

// variable to hold write success or failure
bool success = false;

// attempt non-blocking write from device to pipe until successful
while (!success) SecondPipeInstance::write(data_element, success);
```

#### Blocking Read

The host pipe read interface reads a single element of given datatype from the host pipe. Similar to write, the read interface on the host takes a SYCL* device queue as a parameter. The device read interface consists of the class method read call with no arguments.

On the host:

```c++
// blocking read in host code
float read_element = SecondPipeInstance::read(q);
```

On the device:

```c++
// blocking read in device code
int read_element = FirstPipeInstance::read();
```

#### Non-Blocking Read

Similar to non-blocking writes, non-blocking reads add a `bool` argument in both host and device APIs that is passed by reference and returns true in this argument if the read was successful, and false if it was unsuccessful.

On the host:

```c++
// variable to hold read success or failure
bool success = false;

// attempt non-blocking read until successful in host code
float read_element;
while (!success) read_element = SecondPipeInstance::read(q, success);
```

On the device:

```c++
// variable to hold read success or failure
bool success = false;

// attempt non-blocking read until successful in device code
int read_element;
while (!success) read_element = FirstPipeInstance::read(success);
```

#### Host Pipe API with Avalon Streaming Sideband Signals

The following code example instantiates a `StreamingBeat` struct and writes to a pipe.

On the host:

```c++
bool sop = true;
bool eop = false;
int empty = 0;
PipeDataT data = ...
StreamingBeatDataT in_beat(data, sop, eop, empty);
ThirdPipeInstance::write(q, in_beat);
```

The following code example reads from the pipe and extracts the sideband signals.

On the device:

```c++
StreamingBeatDataT in_beat = ThirdPipeInstance::read();
PipeDataT data = in_beat.data;
bool sop = in_beat.sop;
bool eop = in_beat.eop;
int empty = in_beat.empty;
```


### Host Pipe Connections

Host pipe connections for a particular host pipe are inferred by the compiler from the presence of read and write calls to that host pipe in your code. A host pipe can only be connected between the host and a single kernel. That is, all operations to a particular host pipe on the device side must occur within the same kernel. Additionally, host pipes can only operate in one direction (either host-to-kernel or kernel-to-host).

### Testing the Tutorial

In `host_pipes.cpp`, two host pipes are declared for transferring data to the kernel (`PipeIn`) and from the kernel (`PipeOut`).

```c++
using PipeIn = sycl::ext::intel::experimental::pipe<
    InputPipe,       // An identifier for the pipe
    int,             // The type of data in the pipe
    8,               // The capacity of the pipe
    PipePropertiesT  // Customizable pipe properties
    >;

using PipeOut = sycl::ext::intel::experimental::pipe<
    OutputPipe,      // An identifier for the pipe
    int,             // The type of data in the pipe
    8,               // The capacity of the pipe
    PipePropertiesT  // Customizable pipe properties
    >;
```

These host pipes are used to transfer data to and from `KernelCompute`, which reads a data element from `PipeIn`, processes it using the `SomethingComplicated()` function (a placeholder computation), and writes it back via `PipeOut`.

```c++
struct Kernel {
  int count;

  void operator()() const {
    for (size_t i = 0; i < count; i++) {
      auto d = PipeIn::read();
      auto r = SomethingComplicated(d);
      PipeOut::write(r);
    }
  }
};
```

On the host side, we write all the data to `PipeIn`, launch the kernel, and finally read the data back from `PipeOut` to verify it.

## Build the `Host Pipes` Tutorial

>**Note**: When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script in the root of your oneAPI installation every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### On Linux*

1. Change to the sample directory.
2. Build the program for Intel® Agilex® 7 device family, which is the default.
   ```
   mkdir build
   cd build
   cmake ..
   ```
   > **Note**: You can change the default target by using the command:
   >  ```
   >  cmake .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
   >  ```
   >
   > This tutorial only uses the IP Authoring flow and does not support targeting an explicit FPGA board variant and BSP.

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile and run for emulation (fast compile time, targets emulated FPGA device).
      ```
      make fpga_emu
      ```
   2. Generate the optimization report. (See [Read the Reports](#read-the-reports) below for information on finding and understanding the reports.)
      ```
      make report
      ```
   3. Compile and run for simulation (fast compile time, targets simulated FPGA device).
      ```
      make fpga_sim
      ```
   4. Compile and run for FPGA hardware (longer compile time, targets an FPGA device).
      ```
      make fpga
      ```	

### On Windows*

1. Change to the sample directory.
2. Build the program for the Intel® Agilex® 7 device family, which is the default.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```
   > **Note**: You can change the default target by using the command:
   >  ```
   >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
   >  ```
   >
   > This tutorial only uses the IP Authoring flow and does not support targeting an explicit FPGA board variant and BSP.

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile and run for emulation (fast compile time, targets emulated FPGA device).
      ```
      nmake fpga_emu
      ```
   2. Generate the optimization report. (See [Read the Reports](#read-the-reports) below for information on finding and understanding the reports.)
      ```
      nmake report
      ```
   3. Compile and run for simulation (fast compile time, targets simulated FPGA device).
      ```
      nmake fpga_sim
      ```
   4. Compile and run for FPGA hardware (longer compile time, targets an FPGA device).
      ```
      nmake fpga
      ```
> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

#### Read the Reports

1. Locate `report.html` in the `host_pipes.report.prj/reports/` directory.
2. Open the **Views** menu and select **System Viewer**.
3. In the left-hand pane, select **KernelCompute** under the System hierarchy.

In the main **System Viewer** pane, the pipe read and pipe write for the kernel are highlighted in the **KernelCompute.B1** block. Selecting **KernelCompute.B1** in the left-hand pane gives an expanded view of this block in the main pane, with the pipe read represented by a 'RD' node, and pipe write as a 'WR' node. Clicking on either of these nodes gives further information for these pipes in the **Details** pane. This pane will show that the read is reading from `InputPipe`, and that the write is writing to `OutputPipe`, as well as verifying that both pipes have a width of 32 bits (corresponding to the `int` type) and depth of 8 (which is the capacity that each pipe was declared with).

## Run the `Host Pipes` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./host_pipes.fpga_emu
   ```
2. Run the sample on the FPGA simulator.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./host_pipes.fpga_sim
   ```
> **Note**: Running this sample on an actual FPGA device requires a BSP that supports host pipes. As there are currently no commercial BSPs with such support, only the IP Authoring flow is enabled for this code sample.
	
### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   host_pipes.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   host_pipes.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
> **Note**: Running this sample on an actual FPGA device requires a BSP that supports host pipes. As there are currently no commercial BSPs with such support, only the IP Authoring flow is enabled for this code sample.

## Example Output

```
Data:  0
Data:  1
Data:  2
Data:  5
Data:  8
Data: 11
Data: 14
Data: 18
Data: 22
Data: 27
Data: 31
Data: 36
Data: 41
Data: 46
Data: 52
Data: 58

PASSED
```

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
