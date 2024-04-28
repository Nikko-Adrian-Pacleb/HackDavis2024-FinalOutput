import Link from "next/link";
import image from "next/image";
export default function Patients() {
  return (
    <div className="flex flex-col items-centers bg-slate-100 min-h-screen mx-48">
      <h1 className="text-4xl font-bold bg-slate-100 text-blue-300 font-mono px-16 py-16">
        Patients' Images
      </h1>
      <div className="grid grid-cols-3 grid-rows-2 text-center items-center py-14 gap-x-8 gap-y-4 bg-slate-100">
        <figure>
          <Link href="/patients/logs">
            <img src="/angry.png" height={400} width={300} alt="Patient 1" />
          </Link>
          <figcaption className="text-blue-300 text-l font-mono font-bold">
            Patient 1
          </figcaption>
        </figure>
        <figure>
          <img src="/disgust.png" height={400} width={300} alt="Patient 2" />
          <figcaption className="text-blue-300 text-l font-mono font-bold">
            Patient 2
          </figcaption>
        </figure>
        <figure>
          <img src="/happy.png" height={400} width={300} alt="Patient 3"></img>
          <figcaption className="text-blue-300 text-l font-mono font-bold">
            Patient 3
          </figcaption>
        </figure>
        <figure>
          <img
            src="/neutral.png"
            height={400}
            width={300}
            alt="Patient 4"
          ></img>
          <figcaption className="text-blue-300 text-l font-mono font-bold">
            Patient 4
          </figcaption>
        </figure>
        <figure>
          <img src="/sad.png" height={400} width={300} alt="Patient 5"></img>
          <figcaption className="text-blue-300 text-l font-mono font-bold">
            Patient 5
          </figcaption>
        </figure>
        <figure>
          <img
            src="/surprise.png"
            height={400}
            width={300}
            alt="Patient 6"
          ></img>
          <figcaption className="text-blue-300 text-l font-mono font-bold">
            Patient 6
          </figcaption>
        </figure>
      </div>
    </div>
  );
}
